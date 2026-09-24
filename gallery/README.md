# gallery/ — the notebook figure estate, as an interactive gallery

Two things live here, in three subfolders:

1. **A census** of every figure the repo's notebooks draw, classified against
   the chart taxonomy at <https://www.react-graph-gallery.com/>.
2. **A React + D3 app** that rebuilds the recurring chart families as
   interactive charts over the real filter dataset, so a question like
   "does iron track EC at Addis but not Beijing?" is a dropdown rather than a
   new notebook cell.

```
gallery/
├── census/                 # what the notebooks already draw
│   ├── build_census.py     # scans every .ipynb, classifies each figure cell
│   ├── CENSUS.md           # human-readable rollup
│   ├── chart_census.json   # one record per figure cell
│   └── chart_census.csv    # same, flat
├── data/                   # repo data -> JSON the browser can read
│   ├── export_data.py      # the exporter (reads the pickle via scripts/)
│   ├── export_calibration.py       # calibration_explorer batch results -> calibration.json
│   ├── export_calibration_runs.py  # explorer per-filter readouts + diagnostics -> calibration_runs.json
│   ├── export_similarity.py        # frozen AIRSpec/VIBES spectra -> similarity/ (Spectral similarity tab)
│   ├── export_meeting_followup.py  # 17 Sep meeting follow-up tables -> meeting/ (Meeting 17 Sep follow-up tab)
│   └── fetch_basemap.py    # one-off Natural Earth download for the map
└── app/                    # the React + D3 gallery
    ├── public/data/        # generated JSON — committed, so the app just runs
    └── src/
        ├── charts/         # one subfolder per react-graph-gallery category
        │   ├── correlation/    Scatterplot (combined / per-site grid; colour or size by a
        │   │                   third variable; brush to select), DensityHexbin,
        │   │                   ConnectedScatter, Correlogram (r matrix / scatter matrix),
        │   │                   SpeciesGraph (network / arc / dendrogram), AvailabilityHeatmap
        │   ├── distribution/   DistributionPanel (box / box+points / violin / beeswarm),
        │   │                   Histogram (overlay / density / small multiples / mirror
        │   │                   with two-sample statistics), Ridgeline
        │   ├── evolution/      Timeseries (overlay / per-site, synchronized date cursor),
        │   │                   MonthlyBand, SeasonalClock (circular barplot), MonthlyHeatmap
        │   ├── ranking/        CorrelationRanking (lollipop), Barplot, SiteRadar,
        │   │                   ParallelCoordinates (brushable axes)
        │   ├── partOfWhole/    CompositionTreemap (treemap / circle pack / donut / stacked
        │   │                   bars, family → species hierarchy)
        │   ├── map/            SiteBubbleMap
        │   ├── flow/           MethodSankey (site → EC method → HIPS / season)
        │   ├── pmf/            SourceStack (stacked / streamgraph), SourceSeasonality
        │   └── calibration/    the grid: SpecCurve, CutoffSweep, KSweep, SlopeInterceptTrap,
        │                                 SpectraDumbbell, CrossSiteHeatmap
        │                       one run: CVCurve, RunCrossplot, RunSeries, SplitCheck
        │                       cross-site: CrossSiteTransfer (every site under one calibration,
        │                                   transfer table, site ordering across calibrations)
        │                       cohorts: CohortComposition, SelectionRanking, OverlapMatrix
        │                                (matrix / chord), SpectraOverlay, AnalogLab, HipsYork
        ├── components/     ChartFrame (+ SVG/PNG export), SubsetBar, DateBrush, Legend,
        │                   ColorLegend (continuous ramp + bubble-size key), FieldSelect,
        │                   Axes, SampleDrawer
        ├── hooks/          useGalleryData (loading + responsive size), useTooltip
        ├── lib/            stats (OLS/Deming/KDE/box), highlight (cross-chart hover/pin),
        │                   axes (shared crossplot axis policy), url (hash state), export,
        │                   derived, theme, types
        ├── pages/          CensusPage, CalibrationPage
        └── styles.css      the one stylesheet for everything that isn't an SVG mark
```

## Running it

```bash
cd gallery/app
npm install
npm run dev          # http://localhost:5178
```

(From the Claude desktop app the same server is registered as `gallery` in
`.claude/launch.json`.)

The core JSON in `app/public/data/` (census, filters, meta, pmf, calibration,
calibration_runs, world) is committed, so the Census and Calibration pages run
without touching the pickles. The larger study exports are **gitignored** and
must be regenerated locally before their pages load: `baseline_*`,
`spec_curve.json`, `meeting/` and `similarity/` (see the second block below).
Regenerate after the data or the notebooks change:

```bash
python gallery/census/build_census.py     # re-scan notebooks -> census
python gallery/data/export_data.py        # re-export filter data -> app JSON
python gallery/data/export_calibration.py # calibration explorer batch results -> calibration.json
~/anaconda3/bin/python gallery/data/export_calibration_runs.py  # explorer per-filter readouts (needs the explorer's env)
python gallery/data/export_spec_curve.py  # every scored Addis specification -> spec_curve.json
# meeting follow-up: run the two workflows first (~3 min + ~45 min on 12 threads), then export
uv run --no-sync python research/ftir_hips_chem/workflows/run_meeting_followup_20260917.py
uv run --no-sync python research/ftir_hips_chem/workflows/run_meeting_grid_20260917.py
uv run --no-sync python gallery/data/export_meeting_followup.py
python gallery/data/fetch_basemap.py      # only if world.geojson goes missing
```

Gitignored study exports (each reads results the matching research workflow
wrote under `research/*/output/`, so run that workflow first if they are absent):

```bash
uv run python gallery/data/export_baseline_comparison.py      # baseline_comparison.json + pool/addis/paired CSVs
uv run python gallery/data/export_baseline_applicability.py   # baseline_applicability.json
uv run python gallery/data/export_baseline_external_pilot.py  # baseline_external_pilot.json
uv run python gallery/data/export_baseline_followup.py        # baseline_followup.json, *_followup_*.csv, notebooks
uv run python gallery/data/export_similarity.py               # similarity/*.bin + similarity.json
uv run python gallery/data/export_vibes_fullrange_traces.py   # similarity/traces_VIBES-full.*
uv run python gallery/data/export_etad_ids.py                 # similarity/etad_ids.json
```

## What the census found

211 notebooks, 184 of which draw something: **827 figure-producing cells**.

| Category | Figures | Share |
|---|---:|---:|
| correlation | 478 | 58 % |
| evolution | 179 | 22 % |
| distribution | 89 | 11 % |
| ranking | 81 | 10 % |

The estate is overwhelmingly crossplots (281 scatter + 123 colour-by-third-
variable). There is almost nothing in *part of a whole* (1 stacked area,
2 pie calls) and nothing at all in *flow* — not because the questions aren't
there but because matplotlib makes those awkward. Full breakdown in
[CENSUS.md](census/CENSUS.md).

## Conventions this app inherits from the repo

It reads `research/ftir_hips_chem/scripts/` rather than restating anything, so
the gallery and the matplotlib figures cannot drift apart:

- **Site and season colours** come from `config.SITES[...]['color']` and
  `config.ETHIOPIA_SEASONS`; the per-site local calendars (the default season
  filter) are in `app/src/siteSeasons.ts`, sourced in docs/site-seasonality.md.
- **MAC** is `config.MAC_VALUE`. `HIPS Fabs` stays in Mm⁻¹; the µg/m³ form is
  exported under the unambiguous name `HIPS BC`, never as a same-named column
  that silently differs by a factor of 10.
- **Exclusions** come from `outliers.EXCLUDED_SAMPLES` via
  `apply_exclusion_flags`. They are flagged, not dropped: the header toggle
  puts the 3 excluded samples back so you can see what was removed and why.
- **SPARTAN carbon is FTIR-derived.** The public `ChemSpec_EC_PM2.5` and
  `ChemSpec_OC_PM2.5` values are labelled `EC/OC (ChemSpec FTIR)` in the app.
  They are two-decimal reports of the same in-house FTIR-derived products, not TOR measurements
  or independent EC references. TOR in the calibration and AIRSpec/VIBES tabs
  belongs to the separate IMPROVE calibration/test filters. The site and season
  chips on the main gallery count SPARTAN filters, regardless of field coverage.
- **Deming.** Every crossplot that draws a 1:1 line also reports an
  errors-in-variables slope, per AGENTS.md. The implementation reproduces the
  published numbers exactly — ETAD Fabs/MAC vs FTIR EC, n=190: OLS 1.8983,
  Deming 2.3790, +25.3 %.
- **R² for ranking.** The lollipop chart ranks by R² and carries the sign of
  *r* in colour, never in bar length.

## The subset bar

Every tab except the census carries one set of global controls, and all the
charts below it read from that single subset rather than re-picking per chart.
**The bar renders to match what the active tab's data can support** — a control
that would do nothing is absent, not disabled:

- **Axis fields.** The correlation tab has one x/y pair (with swap) that the
  scatterplot, hexbin, connected scatter and coverage heatmap all read; every
  other filter tab has one *measurement* that its charts share. Previously each
  chart kept its own picker and the hexbin's "same pair as the scatterplot"
  claim was only true by coincidence.
- **Crossplot axes** (correlation tab) — one policy for the scatterplot, hexbin
  and connected scatter together: the 1:1 line (squares the panels, shares the
  domain, turns on Deming), an axes mode — `Data` (0 always visible, negatives
  kept), `From 0,0` (both axes start at zero; points below are not drawn but
  stay in the fit, and the chart says how many), `Show intercept` (extend the
  axes to the fitted lines' y-intercepts and mark them) — and log axes. Fit
  lines are clipped to the axes so a slope above 1 no longer leaves through
  the top margin.
- **Date range** — a brushable timeline of filters per month, stacked by site.
  Drag to keep a range; it snaps to whole months and every tab respects it.
  Beijing reaches back to 2013 and Delhi starts mid-2022, so "only the overlap"
  used to mean editing a notebook.

- **Season calendar** — `dry_feb` (the repo default) or `belg_feb` apply
  shared Ethiopian month bins across sites. Select one site to enable its
  source-backed local calendar; PMF offers the Addis local calendar because its
  factor solution is Addis-only. Switching calendars relabels and filters by
  month without re-exporting. The sources and current per-site filter counts
  are in [site-seasonality.md](../docs/site-seasonality.md).
- **Seasons** and **sites** — click to include/exclude.
- **Excluded samples** — off by default (`get_clean_data`), toggle to put the
  3 back.

Each chip carries its own count, computed *before* that dimension's filter is
applied, so a chip always shows what selecting it would give you. The counts
cross-update. Under the default per-site calendar the season chips are grouped
by site (Beijing 73/84/100/118, Delhi 7/45/22/22, JPL 163/115, Addis 55/69/66);
under the shared Ethiopian `dry_feb` bins, with Belg selected the site chips read
84/45/70/51 (= the 250 in the Belg chip). The filters live in a floating side
panel opened from the tab on the left edge. A chip whose count is zero is struck through and
disabled rather than silently yielding an empty chart — though no combination
in the current data reaches zero, so that path is unexercised.

On the **PMF sources** tab the site chips and the excluded toggle are gone
entirely, because the factor solution is ETAD-only and carries no exclusion
flags; the bar says so and the count switches to "102 of 102 PMF filters".
The per-tab capability table is `CAPS` in `App.tsx`.

Why the calendar is a control and not a caption: February is a high-BC month
and the two published Ethiopian calendars assign it to different seasons, so
seasonal means are genuinely not interchangeable. Measured in the app, the dry
cohort is **358 filters under `dry_feb`** but **293 under `belg_feb`** — the
same 65 February filters landing on either side. Say which convention you used
whenever you report a seasonal statistic; `ETAD_Factor_Analysis.ipynb` is the
one notebook on `belg_feb`.

The correlogram and the composition treemap are computed from the live subset
in the browser, not read from a precomputed file — a precomputed copy would
silently ignore these filters. `correlation.json` and `composition.json` were
removed for exactly that reason.

Per-chart *site* selects are gone from every chart except the lollipop (where
pooling sites manufactures correlation out of site offsets, so "one site" has
to stay an explicit choice). They used to fight the site chips: deselect Addis
in the bar, pick Addis in the chart, get an empty panel. Where a chart needs
per-site separation it now has a **layout** control — `Combined` / `Per site`
on the scatterplot, `Overlay` / `Per site` on the timeseries — mirroring
`PlotConfig.layout` in the repo's plotting package, with the axis domain shared
across panels so slopes stay comparable.

## State lives in the URL

Tab, calendar, seasons, sites, date range, excluded toggle, axis fields and a
pinned filter are all in the hash:

```
#tab=correlation&x=HIPS+BC&y=EC+(FTIR)&sites=Addis+Ababa&from=2023-01-01&to=2023-12-31&pin=ETAD-0088
```

Copy the link into a message or a notebook and it reopens exactly that view.
The back button and a pasted link both apply.

## Linked views: hover, click, and the sample drawer

Hovering a filter in any point-based chart (scatterplot, beeswarm, timeseries)
enlarges the same filter in every other chart on the tab and dims the rest.
The scatterplot and the analog lab's PCA use closest-point detection
(`d3.Delaunay` over the drawn points, one hit rectangle) rather than a mouse
handler per circle, so a 2 px mark in a dense cloud answers the pointer from
18 px away. The timeseries carries a **synchronized cursor**: moving across
any panel drops the same *date* on every other panel with each series'
nearest value labelled, which is how "was Delhi high the same month" gets
read off per-site panels whose time axes differ.
**Clicking opens the sample drawer** and pins the filter (it stays highlighted
on every tab, with a summary card in the subset bar). The drawer shows:

- every measurement the filter carries, grouped by family, with **where it
  sits within its site**: a range strip with the site's IQR, median tick and
  this filter's dot, plus the site percentile;
- the ratios worth reading off one filter (HIPS BC/FTIR EC, OC/EC,
  OM/OC, EC/PM2.5) against the site median of the same ratio;
- **a mini crossplot with its own axis pickers** — any pair, same site or all
  sites — with this filter ringed and its residual from the OLS fit stated;
- its neighbours in time at the site, click-through, plus ← / → keys to step.

This is the question a static figure can never answer: "which point in the
crossplot is that spike in the timeseries, and what else is odd about it?"

Legends are interactive: click a series to drop it from the chart (the
react-graph-gallery "inline legend" pattern). Anything coloured by a number
(the hexbin, the cross-site heatmap, the overlap matrix, the scatterplot's
`Value` mode) draws its ramp through one `ColorLegend` component that samples
the chart's own scale, so the legend cannot disagree with the marks; the
bubble map and the scatterplot's size channel use the matching `SizeLegend`.

## Patterns lifted from react-graph-gallery.com

A 2026-09-21 walk of the gallery's 41 chart pages against this app, and what
it added. The census's second-largest family is 123 crossplots that colour
by a third measured quantity ("iron colour-coded IR BCc vs FTIR EC"), which
had no gallery form:

- **Scatterplot `Value` colour + `size`** — a third measurement on a
  sequential ramp (2–98 % of its range so one outlier doesn't wash the rest
  out) and a fourth in the mark area (sqrt scale, 98 % clamp). Iron is the
  default third variable, per the census.
- **Histogram layouts** — `Overlay`, `Small multiples` (one panel per site,
  shared axes) and `Mirror` (any two groups, site or season, back to back),
  with the bin count on a slider because the bin size changes what a
  histogram appears to say.
- **Box + points** — the boxplot-with-jitter variant: the five-number summary
  over every filter it summarises.
- **Spectra-space dumbbell** (calibration) — each configuration's readout on
  raw, AIRSpec and second-derivative spectra joined so the *change* is the
  mark; every "raw vs AIRSpec" table in the write-ups, in one chart.
- **Chord view of cohort overlap** — the same shared-membership rows as the
  matrix, as ribbons; the first chart in the gallery's *flow* category that
  the estate had never drawn.
- **Transitions** — every chart's SVG carries the `animated` class, so a
  subset, field or encoding change moves marks rather than snapping (CSS
  transitions on presentation attributes; no animation library).

Second pass, same day, the chart types the site has that the estate never
drew, each over the live subset:

- **Brush to select** (scatterplot toggle) and **brushable axes** (parallel
  coordinates) write one shared set of filter ids (`selected` in
  `lib/highlight`). Every point-based chart on any tab dims what is not in the
  set; the subset bar shows "N brushed" with a clear button. It is a
  highlight, not a filter: fits and counts still use the whole subset.
- **Scatter matrix** — the gallery's actual correlogram, as an encoding of
  the Correlogram panel: up to six species of a family (or the six best
  covered), histograms on the diagonal, OLS line and r in each pair, points
  linked to hover and the drawer.
- **Species network · arc · dendrogram** — the same r matrix as a graph.
  Network and arc draw only the pairs above an `|r|` threshold; the
  dendrogram is average-linkage clustering on 1 − |r| with a cut line that
  colours the clusters. The first *flow*-category and hierarchy charts over
  chemistry rather than cohorts.
- **Parallel coordinates** — every species of a family, every filter, one
  polyline each, per-axis brushes intersected. What the radar shows as site
  medians, per filter.
- **Seasonal clock** — the circular barplot: monthly medians around the year
  with the active calendar's seasons shaded, so Dec–Jan are neighbours and
  February's move between calendars is visible.
- **Monthly heatmap** — site × year-month coloured by the measurement (the
  coverage heatmap on the correlation tab counts filters; this one shows the
  value), with a season strip and a continuous legend.
- **Composition hierarchy** — the treemap and donut gained a family → species
  level, plus **circle pack** and **stacked bars** (the one encoding that
  compares sites on one axis).
- **Density layout** and **mirror statistics** on the histogram: the mirror
  reports median difference, Welch t and p, Cohen's d and the Mann–Whitney
  AUC, effect sizes first (`lib/stats.twoSample`, checked against hand
  calculations).
- **Direct labels** at the end of each line in the timeseries overlay.

## Export

Every chart panel has **SVG** and **PNG** buttons. SVG is the editable form for
Illustrator/Inkscape; PNG is rendered at 2× for a slide. The serialiser adds a
white background and the app font so the file looks the same outside the page.
The decks in this repo are built from figures, so a chart that could only be
screenshotted was a chart that would be redrawn in matplotlib.

## PMF source apportionment

The **PMF sources** tab reads the ETAD factor solution (102 filters,
Jan–Dec 2023) through `etad_factors.load_etad_factors_with_filter_ids()` →
`normalize_gf_fractions()` → `add_dominant_source()`.

That normalisation step is not optional. Raw `GF1`–`GF5` are PM2.5 *mass*
fractions summing to 0.03–0.46 per row, not relative source contributions —
stacked directly they are wrong, `dominant_fraction` tops out near 0.24, and no
filter ever crosses a 30 % threshold. After normalisation the mean dominant
fraction is **0.462**, which the exporter prints on every run as a check.

Five sources: charcoal, wood burning, fossil fuel, polluted marine, sea salt
mixed. `pmf.json` carries both the normalised fractions and the absolute
`K_Fn` µg/m³ contributions, so the stacked area can switch between a
100 %-relative mix and apportioned mass. Each source ships an explicit `key`
matching what `add_dominant_source` writes, so the app never has to match a
source by a prefix of its display label.

This tab is also where the two emptiest gallery categories get filled: the
estate has exactly **one** stacked-area figure in 827, and source
apportionment is the question it answers.

## Units

The source table mixes units: trace metals are stored in **ng/m³** while ions
and carbon are in **µg/m³**. The exporter reads the declared
`Concentration_Units` column and converts everything to µg/m³ (30 columns
converted), rather than relying on `prep.to_ugm3`'s median>100 heuristic. Units
per field travel in `meta.json` and appear in every axis label and dropdown.

Sanity check after conversion — speciated mass as a fraction of measured PM2.5:
Beijing 56 %, Delhi 71 %, JPL 75 %, Addis 79 %. Before the fix, silicon alone
read 906 µg/m³ against a 41 µg/m³ PM2.5 total.

## Calibration tab

The phase-3 FTIR-EC calibration and optimisation work, read out of the
calibration explorer's batch results (`calibration_explorer/cache/
batch_results.jsonl`, 82,934 configuration × site × k rows). Nothing is
recomputed: `export_calibration.py` keeps the rule-k row per configuration ×
evaluation site (11,190 rows) and the full k sweep for Addis under protocol A
(14,092 rows), plus the dated write-ups' first paragraphs. Four charts follow
the argument in those write-ups:

- **Cutoff sweep** — readout vs cohort size, one line per calibration spectra
  space, locked cutoff marked. The 2026-08-20 dense sweep's basin at
  ocec-440–490 × AIRSpec is visible here; the coarse ladder never looked below 600.
- **k sweep** — intercept, slope, R² and held-out TOR R² against component
  count for one configuration, rule k marked: the intercept keeps shrinking
  with k while the guardrails deteriorate.
- **Slope vs intercept** — every configuration as a point, the slope box
  shaded, guardrail failures dimmed, top 5 by score numbered. The full-scale
  slope trap in one picture.
- **Spectra-space dumbbell** — one row per cohort × cutoff × selection space,
  the same configuration on raw / AIRSpec / second-derivative spectra joined.
  `Locked cutoffs` is the setup matrix; `One cohort, every cutoff` is the
  dense sweep, row by row.
- **Cross-site heatmap** — the best Addis configurations at all five SPARTAN
  sites: blue at Addis and Delhi, white at Beijing and Pasadena is the
  compositional-offset signature.

One configuration is selected across the page (click a point or a heatmap
row); when it matches an exported preset, the per-filter section below
follows.

### One configuration, every filter

`export_calibration_runs.py` imports the explorer in-process, waits for its
data to load, and calls its own HTTP handlers through Flask's test client, so
every number is exactly the explorer's. It exports nine named configurations
(the six built-in presets, the dense-sweep basin ocec-440 × AIRSpec, ocec-450
× 2nd derivative, and Ethiopia-shaped × AIRSpec) at all five SPARTAN sites,
plus the cohort diagnostics. That gives the explorer's Calibrate, Series,
Target-readout, Selection, Analogs, Sites and HIPS tabs a gallery form:

- **CV curve** — RMSECV vs components with the fold SE band and the rule k.
- **Crossplot + residuals** — predicted EC vs HIPS Fabs ÷ MAC, season
  colours, the explorer's own fit row (estimator, evaluation set, MAC 6/10/17
  are controls), or vs deployed SPARTAN EC on the same filters. Addis and
  Beijing points open the sample drawer.
- **Dated series** — with the 45-day rolling median, deployed EC hollow, and
  the plausibility card (negative days, days above 8, group medians).
- **Blind-half check** — early/late and odd/even halves, same n.
- **Cohort composition** — OC/EC of the cohort vs the pool, Addis marker.
- **Selection ranking** — metric vs rank, raw and corrected, cutoff marked;
  or the pool distribution with the cutoff value.
- **Cohort overlap** — pairwise shared members as a fraction of the smaller,
  as a matrix or a chord diagram (hover an arc to isolate its ribbons).
- **Cohort spectra / site spectra** — median ± IQR, raw or AIRSpec.
- **Analog lab** — PCA of the pool with the committed analogs and Addis
  starred; committed rank vs each alternative metric with Spearman ρ. The
  PCA axes follow the 0.5–99.5 % bulk: 28 extreme spectra (PC2 ≈ −2 against
  a bulk at ±0.06) used to flatten the whole pool into one pixel row and hide
  its arc; they are pinned hollow at the edge and counted under the chart.
- **HIPS diagnostics** — per-site York fits under three blank-line
  conventions, ± SE, plus the blank ledger.

What is still only in the explorer: any configuration not in the preset list,
manual k, the evaluation-view levers (lot, season, half), the stability
bootstrap, and running a new batch. `python calibration_explorer/app.py` →
port 5058. The gallery is the reading view; the explorer is the lab.

## What it does not cover

- **Aethalometer time series.** The app is filter-based (942 filters, 4 sites).
  The `processed_sites/*.pkl` minute-resolution aethalometer data, diurnal
  profiles and wavelength/AAE work are not exported yet — that is the obvious
  next dataset, and would light up the `flow/` category too.
- **FTIR spectra.** The phase-3 spectral notebooks are the other large family
  with no representation here; a spectrum overlay is a line chart the estate
  genuinely needs.
- **A network view of spectral similarity** (ftir_52's spectral map) and a
  **dendrogram of spectral clusters** (the explorer's k-means-3 sub-types)
  need pairwise similarities or a linkage exported from the explorer; the
  analog-lab export carries ranks and PCA coordinates only. The species graph
  is the same three chart types over chemistry, where the data is in hand.
- **Edge bundling**, **choropleth / hexbin / connection maps**, **cartogram**,
  **wordcloud** and **pie** are not planned: four sites and 28 species do
  not carry them.
- **Canvas rendering** for the largest scatters was deliberately not done:
  every chart is exported as SVG, and a canvas layer would vanish from the
  file.

## Rendering invariants

Charts are checked in-browser for geometry that escapes its SVG canvas
(every `text`/`rect`/`circle`/`path`/`line` outside a `clipPath` against its
`<svg>` box); all tabs render clean as of 2026-09-21. Two classes of bug that check caught, worth not
reintroducing:

- **Never floor an axis domain at zero.** `EC (FTIR)` carries 6 genuinely
  negative values near the detection limit (min −4.0, a Delhi filter).
  `Math.max(0, …)` on the domain drew those points outside the axes entirely.
  Floor at `Math.min(0, dataMin)` instead.
- **Size gutters from the content.** Ridgeline row labels
  ("Belg (Feb-May, short rains)") and the lollipop's `0.939 · n=190`
  annotations both overran fixed margins. Both now measure the text.

- **Drop by-construction pairs.** `HIPS BC` is `HIPS Fabs ÷ MAC`, so their
  correlation is 1.000 by arithmetic. `lib/derived.ts` lists such pairs; the
  correlogram omits them and the lollipop draws them hollow and grey rather
  than letting a unit conversion top the ranking.

Histogram and ridgeline clip to 1–99 % by default so a long right tail doesn't
squeeze the bulk into a few pixels — and both state how many values fall
outside rather than dropping them silently. The ridgeline's clip is a toggle.

## AIRSpec / VIBES comparison

Open `http://localhost:5178/#tab=baseline` for the completed paired baseline
comparison. Regenerate its separate export with:

```bash
uv run --locked --no-sync python gallery/data/export_baseline_comparison.py
```

The page uses frozen Colab predictions and the subsequent audit. It provides
full/restricted cohort selection, site/loading display subsets, RMSE/MAE/bias/
predictive R², residual plots, paired loading-band intervals, 12 individual
inspection spectra, blank/injection diagnostics and downloadable inclusion
ledgers. Display subsets never refit the models or change the frozen scores.
The complete-cohort interval and loading-band chart remain explicitly labelled
when a display subset is selected. All negative predictions remain in the errors.

**Spectral-cut distinction:** the September 10 experiment excluded
1800–2500 cm⁻¹, optionally also >3600 or >3500, from **analog matching only**.
All its PLS fits retained the complete 2002-channel AIRSpec grid. The paired
Colab benchmark does not repeat masked analog selection: it compares the full
pool and the earlier lowest-OC/EC membership. The gallery's amber regions are
an explanatory overlay and do not claim that the displayed predictions use
masked features. Historical membership changes appear separately.

In the spectrum panel, **Excluded regions → Hide excluded regions** hides the
selected channels from the curves, preserves gaps on the true wavenumber axis,
and fits the y-axis to visible values. **Shade excluded regions** restores all
channels. The **VIBES − AIRSpec** spectrum view shows the pointwise difference
between corrected spectra, with zero as the agreement reference. Neither control
recomputes corrections or EC predictions. SVG/PNG exports label the selected mask
and its display-only status.

The paired-error histogram, inspired by the
[React Graph Gallery histogram](https://www.react-graph-gallery.com/histogram),
shows `abs(VIBES − TOR) − abs(AIRSpec − TOR)` for every selected physical filter.
Negative values favor VIBES. All tails remain included, bins share a boundary
at zero, and the mean equals the methods' MAE difference. Bin counts, spectral
display choices and existing filters persist in copied URLs. The counts and
median describe the selected filters; they are not a significance test.

The exporter checks the frozen bundle hashes, original prediction hashes,
paired metric reconciliation and retained physical-filter IDs. It does not
require Drive or repeat any correction/model fit. The 253 Addis target spectra
have no independent thermal EC reference in this comparison. Global gallery
sample controls are hidden on this page because they describe a different dataset.

### Research diagnostics

The baseline page also shows training spectral distance, paired region
contributions, site × EC-loading error differences, and individual held-out
blank residuals. The region contributions are the saved VIBES − AIRSpec
decomposition and reconcile to the paired predictions. The site/loading matrix
is a retrospective screen using measured IMPROVE test EC, not a fitted routing
rule. Cells with fewer than five filters are left uncolored.

Rebuild the spectral-distance export separately with:

```bash
uv run --locked --no-sync python gallery/data/export_baseline_applicability.py
```

It verifies the frozen input hashes and fits an exploratory, eight-component
PCA to each method's 10,066 IMPROVE **training** spectra only. The displayed
T² and reconstruction-error percentile ranks compare 2,327 held-out IMPROVE
spectra and 253 Addis target spectra to those training distributions. These are
not saved PLS scores or validated applicability cutoffs. Addis has no thermal
EC truth here, so its transfer bars report spectral distance only; no Addis
error or accuracy is inferred. The gallery does not display prospective-study
cards. The follow-up analyses are instead in the CLI-executable
`research/ftir_hips_chem/vibes_followup_experiments.ipynb`, with frozen inputs,
saved result tables, and an independent-reference gate for Addis. The separate
`gallery/data/export_baseline_followup.py` exporter validates and publishes its
selection and routing scores under a clearly labelled **executed notebook
follow-up** panel. The original Colab score table remains unchanged. Rebuild it
after executing the notebook with:

```bash
uv run --locked --no-sync python gallery/data/export_baseline_followup.py
```

The gallery's follow-up chart compares all six selections for each correction
method on the same 2,327 IMPROVE test filters. The restricted-membership table
separates its 137 original test members from the 2,190 outside members. The
routing chart shows grouped training-site folds and the previously inspected
outer sites, with the severe VIBES fold error retained. Download links expose
the executed notebook, per-filter predictions, membership and metrics. These
analyses are exploratory because the original test outcomes had already been
inspected. The Addis readiness gate found no independent thermal EC matches,
so no Addis accuracy score appears.

## Spectral similarity tab

The **Spectral similarity** tab answers "what looks most like this?" for any
target: one of the 158 sites in the frozen full-profile AIRSpec / VIBES run
(157 IMPROVE sites plus Addis), or one season at a site. Every other site and
site-season is ranked by the **mean signed Pearson r** over all filter pairs,
excluding self-pairs. The 500 most similar individual filters from other sites
are scored exactly as `seasonal_analogs.mean_correlation_scores` scores an
analog library, and the exporter asserts parity to within 1e-9. The page shows
the ranking, a season-by-season matrix for the target's site, the target's
spectra over its closest matches, where those matches come from by site and
season, and a downloadable filter list. Spectra method (AIRSpec / VIBES) and
all four analog-selection masks are switchable. The masks change which
channels are compared, never the spectra.

Site names, states and coordinates come from the `Sites` sheet of the
IMPROVE Query Wizard workbooks, located by `improve_io.improve_dir()` (set
`AETHMODULAR_IMPROVE_DIR` if the Drive folder isn't mounted); Addis comes from
the frozen ETAD metadata. They appear wherever a site code does: tooltips, the
filter table, and both CSVs. In the spectra chart, hover a trace to identify
it and click to pin it.

Seasons are fixed month bins. Addis uses the canonical Dry Oct–Feb, Belg
Mar–May and Kiremt Jun–Sep; IMPROVE sites use meteorological seasons. Six
Addis filters have no date in the frozen ledger and count toward all-year Addis
only. Rebuild with:

```bash
uv run --locked --no-sync python gallery/data/export_similarity.py
```

The export takes about ten seconds and writes about 27 MB to
`app/public/data/similarity/`. Each method × mask pair has a group × group
score triangle and a top-500 list, and each method has binned per-filter
traces. The page loads only the files for the current method and mask.
