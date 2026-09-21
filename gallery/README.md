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
│   └── fetch_basemap.py    # one-off Natural Earth download for the map
└── app/                    # the React + D3 gallery
    ├── public/data/        # generated JSON — committed, so the app just runs
    └── src/
        ├── charts/         # one subfolder per react-graph-gallery category
        │   ├── correlation/    Scatterplot (combined / per-site grid; colour or size by a
        │   │                   third variable), DensityHexbin, ConnectedScatter,
        │   │                   Correlogram, AvailabilityHeatmap
        │   ├── distribution/   DistributionPanel (box / box+points / violin / beeswarm),
        │   │                   Histogram (overlay / small multiples / mirror), Ridgeline
        │   ├── evolution/      Timeseries (overlay / per-site, synchronized date cursor), MonthlyBand
        │   ├── ranking/        CorrelationRanking (lollipop), Barplot, SiteRadar
        │   ├── partOfWhole/    CompositionTreemap (treemap/donut)
        │   ├── map/            SiteBubbleMap
        │   ├── flow/           MethodSankey (site → EC method → HIPS / season)
        │   ├── pmf/            SourceStack (stacked / streamgraph), SourceSeasonality
        │   └── calibration/    the grid: CutoffSweep, KSweep, SlopeInterceptTrap,
        │                                 SpectraDumbbell, CrossSiteHeatmap
        │                       one run: CVCurve, RunCrossplot, RunSeries, SplitCheck
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

The JSON in `app/public/data/` is committed, so the app runs without touching
the pickles. Regenerate after the data or the notebooks change:

```bash
python gallery/census/build_census.py     # re-scan notebooks -> census
python gallery/data/export_data.py        # re-export filter data -> app JSON
python gallery/data/export_calibration.py # calibration explorer batch results -> calibration.json
/Users/ahmadjalil/anaconda3/bin/python gallery/data/export_calibration_runs.py  # explorer per-filter readouts (needs the explorer's env)
python gallery/data/fetch_basemap.py      # only if world.geojson goes missing
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
  `config.ETHIOPIA_SEASONS`.
- **MAC** is `config.MAC_VALUE`. `HIPS Fabs` stays in Mm⁻¹; the µg/m³ form is
  exported under the unambiguous name `HIPS BC`, never as a same-named column
  that silently differs by a factor of 10.
- **Exclusions** come from `outliers.EXCLUDED_SAMPLES` via
  `apply_exclusion_flags`. They are flagged, not dropped: the header toggle
  puts the 3 excluded samples back so you can see what was removed and why.
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

- **Season calendar** — `dry_feb` (the repo default) or `belg_feb`. Both come
  from `config.SEASON_CONVENTIONS`; the app derives each filter's season from
  its month at render time, so switching relabels every chart with no
  re-export. The bar always names which season February falls in under the
  active calendar.
- **Seasons** and **sites** — click to include/exclude.
- **Excluded samples** — off by default (`get_clean_data`), toggle to put the
  3 back.

Each chip carries its own count, computed *before* that dimension's filter is
applied, so a chip always shows what selecting it would give you. The counts
cross-update: with Belg selected the site chips read 84/45/70/51 (= the 250 in
the Belg chip), and with Addis selected the season chips read 73/51/66 (= the
190 in the Addis chip). A chip whose count is zero is struck through and
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
- the ratios worth reading off one filter (FTIR/TOR EC, HIPS BC/EC, OC/EC,
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
- **Parallel coordinates** and **circular barplot** from the gallery's ranking
  category are the obvious next two; neither is in the estate. Parallel
  coordinates would show all 28 species per filter at once, which the radar
  only does as site medians.
- **A network view** of spectral similarity (the gallery's network chart,
  ftir_52's spectral map) needs pairwise similarities exported from the
  explorer; the analog-lab export carries ranks and PCA coordinates only.
- **Dendrogram** of spectral clusters (the explorer's k-means-3 sub-types)
  likewise waits on an export of the linkage.
- **Canvas rendering** for the largest scatters was deliberately not done:
  every chart is exported as SVG, and a canvas layer would vanish from the
  file.
- **Brush-to-select on the scatterplot** (select a cloud of points and see
  them highlighted on the timeseries) is the next step after hover/pin.

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
