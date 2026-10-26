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
│   ├── export_meeting_followup.py  # 17 Sep meeting follow-up tables -> meeting/ (Follow up meeting view)
│   ├── export_satoshi_followup.py  # Satoshi's 2 Oct questions -> satoshi/ (Satoshi 2 Oct view in Meeting)
│   ├── export_oct1_remaining.py    # rest of the 1 Oct list (blank banks, groupings) -> satoshi/ (Satoshi 2 Oct view in Meeting)
│   ├── export_blanks.py            # PTFE blanks behind the VIBES background -> blanks/ (Blanks tab)
│   └── fetch_basemap.py    # one-off Natural Earth download for the map
└── app/                    # the React + D3 gallery
    ├── public/data/        # generated JSON — committed, so the app just runs
    └── src/
        ├── charts/         # one subfolder per react-graph-gallery category
        │   ├── correlation/    Scatterplot (combined / per-site grid; colour or size by a
        │   │                   third variable; brush to select), DensityHexbin,
        │   │                   ConnectedScatter, Correlogram (r matrix / scatter matrix),
        │   │                   SpeciesGraph (groups as a network / tree), AvailabilityHeatmap
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
        ├── pages/          CensusPage, CalibrationPage, DashboardPage (full-window tiles for one site,
        │                   compare sites or the network; chart kinds in components/dashboard/registry,
        │                   layout in lib/dashboard, charts fit a tile through lib/chartFit; a new chart
        │                   joins the Add chart menu with registerDashboardChart)
        └── styles.css      the one stylesheet for everything that isn't an SVG mark
```

## Running it

**Sites → Data availability** incorporates the existing SPARTAN product and
HIPS coverage notebooks, four-site FTIR/HIPS/aethalometer day overlaps,
sampling cadence and gaps, the filter inventory, and the historical
SPARTAN–IMPROVE comparison. Repackage its sources with
`uv run python gallery/data/export_availability.py` after `uv run aeth doctor`.
This reuses saved network tables and recomputes study coverage through the
shared loaders; it does not download or refresh remote observations.
See [the source and counting notes](../docs/gallery-data-availability.md).

The bubble map offers **Four focus sites** (Beijing, Delhi, JPL and Addis Ababa),
**Single focus site** (with a picker limited to those four), **All SPARTAN sites**,
and **All IMPROVE sites**. Focus-site maps retain the season, date and exclusion
choices while selecting sites independently of the sidebar. IMPROVE uses its
own measurement picker (TOR EC/OC, absorption, mass, iron and soil) and the
locally available canonical cleaned IMPROVE population. Sample-count bubbles
count all samples in that population, independently of measurement coverage.
The compact, committed `improve_map.json` contains site summaries rather than
individual records; regenerate it with
`uv run python gallery/data/export_improve_map.py` after updating IMPROVE data.
Map scope, focus, measurement and size choices survive in a copied URL.
The validated IMPROVE refresh uses yearly FED Aerosol ZIPs for 2003–2025
and a current-year Query Wizard report. Preliminary `IMPAERPRE` exports stay
outside the gallery cache. The map displays the release dates and any known
absorption coverage gaps. Its sample counts still describe the paired,
positive EC/absorption population, rather than every downloaded measurement.

To refresh, export **IMPROVE Aerosol** (validated) in Query Wizard with all
sites, parameters and fields; the current year and all months; text output,
normalized rows, pipe delimiter, and data plus metadata. Then run:

```bash
uv run aeth doctor
uv run python research/ftir_hips_chem/workflows/refresh_improve_gallery.py --refresh-downloads --query "$IMPROVE_VALIDATED_REPORT"
uv run python gallery/data/export_improve_map.py
uv run python gallery/share/build_share.py
```

Set `IMPROVE_VALIDATED_REPORT` to the downloaded file. Raw downloads, flagged
sample Parquets, checksums, monthly coverage and the comparison with the old
cache live under `research/ftir_hips_chem/output/tables/improve_validated_archive/`.
The previous cache is backed up before replacement. Module A and C flow and
duration records remain separate; missing historical volumes stay missing.

Every scope supports dragging to pan, wheel/pinch zoom and +/− buttons.
IMPROVE opens on the US mainland; **All sites** fits the entire available
network, including Alaska, Hawaii and overseas sites. Bubble radii stay fixed
on screen so zooming does not change the size legend.
The SPARTAN network view uses the local site inventory and public ChemSpecPM25
files, with hollow markers where the selected measurement is unavailable.
The four focus sites keep their canonical colours and names in that network
view. Regenerate its compact summaries with
`uv run python gallery/data/export_spartan_map.py`.

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
# Satoshi's 2 Oct questions (analog overlap, swap test, VIP, fit ladder, envelope): ~5 min, then export
uv run --no-sync python research/ftir_hips_chem/workflows/run_satoshi_followup_20261002.py
uv run --no-sync python gallery/data/export_satoshi_followup.py
# Rest of the 1 Oct list (blank-bank runs on Drive, fingerprint groupings): ~5 min, then export
uv run --no-sync python research/ftir_hips_chem/workflows/run_oct1_remaining_20261007.py
uv run --no-sync python gallery/data/export_oct1_remaining.py
# Blanks tab (reads the bundle, the capped run and the Drive blank-bank runs): ~2 min
uv run --no-sync python gallery/data/export_blanks.py
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
uv run python gallery/data/export_vibes_fullrange_traces.py --rank 70  # similarity/traces_VIBES-full-r70.*
uv run python gallery/data/export_vibes_fullrange_results.py  # vibes_fullrange_results.json
uv run python gallery/data/export_etad_ids.py                 # similarity/etad_ids.json
```

### Define it once

Before adding a constant, picker or chart to a page, use the shared one
(full list and what is still open: `docs/gallery-chart-standards.md` §4b):
colours, regions and trace styles in `lib/theme.ts`; names and axis titles in
`lib/labels.ts`; number formats in `lib/format.ts`; season names and the two
Addis calendars in `lib/seasons.ts`; every data file through `lib/data.ts`
(`useExport`, `loadJson`) and spectra through `lib/similarity.loadTraces`;
`components/Pickers`, `Crossplot`, `Key`, `SpectralRegions`, `IntervalStrip`
and `ExportState` for the recurring UI. URL keys must be unique across pages
(development builds warn when two controls share one).

Four chart types cover most research figures; draw with them instead of
hand-rolling d3 and axes: `Crossplot` (1:1), `SpectrumPlot` (reversed
wavenumber axis, faint traces, emphasised lines, median/IQR bands, regions,
hover that names a trace or reads out every line, pin linked to the page
through `highlight={useHighlight()}`), `SweepPlot` (a metric against k, a
cutoff, a rank, a stage or a date: bands or whiskers, the agreement line,
labelled markers, the chosen setting ringed, clipped points drawn hollow at
the edge) and `IntervalStrip` (dot-interval rows). `FacetGrid` lays out small
multiples. Each takes a `children` render prop for the odd extra mark, so a
chart that needs one more line does not have to be rebuilt from scratch.

### Declaring a figure or a whole meeting

`components/Figure` draws a chart from a description: the kind and which
columns of a long table (one row per point) to use.

```tsx
<Figure kind="spectrum" data={rows} wn="wn" y="median" lo="q25" hi="q75" series="series" title="…" />
<Figure kind="sweep" data={splits} x="k" y="addis_intercept" series="baseline" facet="group"
        aggregate={{ stat: 'median', interval: 'iqr' }} xLabel="PLS factors (k)" yLabel="Addis intercept" reference={0} title="…" />
```

Kinds: `crossplot`, `spectrum`, `sweep`, `intervals` (median and interval per
row and series, summarised in the browser from repeated splits), `histogram`
(densities on shared, optionally log, bins), `heatmap`, `table`, and `custom`
for anything a description cannot say (linked charts, live refits): a
component registered with `registerFigure(name, Component)`. `choose` adds a
control that keeps the rows with the picked value of a column; `where`
filters rows; colours come from the figure, then the meeting bundle, then the
gallery's baseline colours.

A meeting page needs no TSX:

1. An exporter, `gallery/data/export_meeting_<date>.py`, that builds long
   tables and the answer text from the saved outputs and calls
   `meeting_bundle.write_bundle(id, tables=…, text=…, colors=…)`. It writes
   `public/data/meetings/<id>.json`. Write the text from the same numbers the
   figures draw, so prose and charts cannot disagree.
2. A spec, `app/src/meetings/<id>.meeting.ts`: sections of Markdown text
   (`@key` = a paragraph from the bundle) and `FigureSpec`s naming its
   tables. It shows up in the Meeting tab's picker by itself.

`ann-20261009` (the 9 Oct questions from Ann) is the worked example:
`gallery/data/export_meeting_20261009.py` and
`app/src/meetings/ann-20261009.meeting.ts`.

### Networks in the Filters sidebar

The chart tabs (Correlation, Distribution, Evolution, Ranking, Part of a whole) draw a
network chosen at the top of the Filters sidebar. **SPARTAN** is the default and starts on
the four focus sites, which keep their `filters.json` rows (HIPS, FTIR groups, exclusions,
local calendars), so the default view is unchanged; any other SPARTAN site can be added and
brings its public ChemSpec rows from `spartan_reference.json`. **IMPROVE** offers TOR EC/OC,
HIPS absorption, mass, iron, silicon and soil, one file per site (`improve_samples/`, from
`uv run python gallery/data/export_improve_samples.py`, gitignored, ~26 MB), ten at a time
because the SVG charts cannot draw all ~388,000 samples. Sites are picked in a searchable
checkbox list (picked first), with **Focus sites**, **Select all** and **Deselect all**
(which shows nothing), or on a large map (**Map…**). With only focus sites picked the local
calendars are the default; with others, shared three-month bins. Map, Flow and the research
tabs always show the focus sites. URL: `net` (spartan default; `net=focus` keeps the old
focus-only view for links), `ssites` (absent = focus sites, `*` = all, `-` = none, else
codes), `isites` (IMPROVE codes).

Builds for outside readers (`publish_pages.py`, `build_share.py`) set
`VITE_HIDE_EXPLAIN=1`, which removes the header's Explain toggle; local dev keeps it.

The **Experimental** pages (tests from the 22 Oct protocol review, the Adama TOR study) are left out of
the header menu and the ⌘K search in every build: they open only from a direct link
(`#tab=experimental`, a page with `&xp=<key>`, keys in `app/src/experiments/index.ts`), and the menu shows
their group only while one is open. Set `VITE_SHOW_EXPERIMENTAL=1` (e.g. in `app/.env.local`) to list them.

### Publishing

`python3 gallery/share/publish_pages.py` publishes the public site
(https://projects.ahmadjalil.com/aethmodular-gallery/) without building anything here:

1. It packs `gallery/app/public/data` into one archive. Machine paths and account names are
   scrubbed, and it refuses to continue if any remain. The same data always gives the same bytes.
2. It attaches the archive to a release of `ahzs645/aethmodular-gallery` (`data-<hash>`). If that
   data is already there, it skips the upload.
3. It starts that repo's deploy workflow (`gallery/share/pages-deploy.yml`, installed there as
   `.github/workflows/deploy.yml`). GitHub then checks out `gallery/app` at the pushed commit,
   unpacks the data, builds, re-checks and deploys. This takes about a minute.

Linked export folders are included in the snapshot; broken links and directory cycles stop
publication. After deployment succeeds, the gallery repo's README records the exact source
commit, data release and Actions run.

Commit and push the gallery source first: the script refuses uncommitted changes under
`gallery/app` (exports in `public/data` excepted), because GitHub builds the pushed commit.
`--dry-run` packs and checks only. `--branch` is the earlier way (build here, push the built
files to the repo's `main`). It preserves the source-build workflow and installs a second
workflow that deploys the prebuilt files on push. It publishes `build-info.json` with the
exact source commit and keeps the existing Pages workflow mode. This fallback also works
when release-asset uploads or Pages settings changes are unavailable.

### Views, sharing and embeds

**Views ▾** in the header holds built-in presets (`lib/views.ts` PRESETS), the reader's own
saved views (this browser only, localStorage) and two share actions. **Copy link** puts the
whole view (tab, networks, sites, fields, seasons, dates, hand-excluded or kept filters, and
the per-chart controls in the query string) into one `#v=<token>` made with
`@firstform/json-url` (the webforms share codec): a preset left unchanged is a ~30-character
reference, anything else is lz-compressed or raw, whichever is shorter (a view keeping 101
brushed filters: 564 characters against 1,352 for the plain URL). **Copy embed code** gives an
iframe with `?embed=1`, which hides the header and filter panel and adds an "Open in the gallery"
link. Plain hash links still work.

SPARTAN and IMPROVE are a multi-select: with both on, sites from either network share the list,
the map and the charts (HIPS absorption and PM₂.₅ mass are measured by both; EC/OC by FTIR in
SPARTAN and TOR in IMPROVE).

With **Brush to select** on the scatterplot, a bar offers **Keep only these** or **Exclude these**:
hand edits that filter every chart on top of the dataset's exclusions, listed (and clearable) in
the Filters panel and carried in the link (`keep`, `drop`). Picking the field the other axis
already shows swaps x and y, so a crossplot never plots a field against itself.

### Pickers and axes

Every dropdown is the one styled `components/Combobox` (native-select height; groups, long
lists and type-ahead), including the chart controls, field pickers, calendar and Jump to.
Axis titles that show the shared x, y or measurement (crossplots, distribution, evolution,
bar plot) are clickable: they open a searchable field popover that edits the same choice as
the sidebar. **Panel axes** in the sidebar's crossplot row switches per-site panels between
one shared range (the default, so slopes compare by eye) and each panel's own range; with
the 1:1 line on, every panel stays square either way. Rendered text shows chemical
subscripts (PM₂.₅, SO₄, NO₃, …) via `lib/format.chemText`; field names stay plain in data
and URLs.

### Sharing a read-only copy

The full app carries ~126 MB of data, most of it gitignored, so a clone is not
a way to share it. `gallery/share/build_share.py` builds a static copy that
offers only the listed tabs (via `VITE_SHARE_TABS`) and keeps only the data
those tabs fetch: Meeting + Map is ~34 MB.

```bash
python3 gallery/share/build_share.py                       # Meeting + Map
python3 gallery/share/build_share.py --tabs meeting,map,blanks
```

The result in `gallery/share/dist/` (gitignored) uses relative paths and must
be served over HTTP, not opened as a file; the `gallery-share` entry in
`.claude/launch.json` serves it on port 5180. It holds unpublished data, so
it stays local until someone decides where to host it.

The Meeting tab's "all held-out IMPROVE" crossplots need
`pool_test_predictions.csv` from
`research/ftir_hips_chem/workflows/run_pool_test_points_20261009.py`, then
`uv run python research/ftir_hips_chem/workflows/migrate_vibes_gallery_rank70.py export-meeting`.

## What the census found

The **Meeting** tab groups **Follow up meeting** and **Satoshi 2 Oct** under a
Meeting dropdown. Each view retains its charts and controls. Copied links keep
the selected view (`#tab=meeting` or `#tab=meeting&meeting=satoshi`); existing
`#tab=satoshi` links still open Satoshi's follow-up.
The picker is a styled combobox with keyboard navigation. The full-range VIBES
status, Satoshi's "Where we stand, 7 Oct" conclusion and short-answer table appear only with
**Explain** enabled, keeping the presentation view focused on charts.
In Satoshi's view, **Filters → Similarity region** chooses the saved exclusion
range. **600–3600 cm⁻¹ · exclude CO₂ + PTFE** is the default; the alternative
is **500–3600 cm⁻¹ · exclude CO₂**. The sidebar describes the excluded bands.
Copied links preserve explicit region choices. The short-answer table is
labelled as a summary of the original 1 Oct region.
The Satoshi view's **Spectra** panel overlays every frozen Addis test-set filter
in the chosen season with the top 50, 100, 250 or 500 saved IMPROVE analogs.
Ranks use correlation to the median Addis spectrum from the meeting analysis;
they are not recomputed from the display traces or the Spectral similarity
page's mean-pair scores. Baseline and season controls, an outer-test-site
toggle, gaps/shading and hover/click filter details use the shared spectra
renderer. The sidebar region controls the saved cohort and excluded channels;
CO₂ is labelled explicitly as 1800–2500 cm⁻¹. Top-N and site controls affect
the display only, leaving the saved calibration scores unchanged.

The Follow up meeting view displays an Addis coverage table independently of
the regression selection. The current gallery has 190 paired filters; the
frozen test-set evaluation has 233 filters, sharing 189 with the gallery and
adding 44 later filters. Gallery filter ETAD-0248 is absent from the frozen
spectral run. The existing SPARTAN product therefore has 189 complete pairs
within that evaluation, while new predictions have 233. Missing pairs never
become zeros. Re-exporting the meeting data preserves its Addis Deming error
ratio (about 3.3354); it must not silently become the equal-error default.
Regression options control the displayed lines and equations, with Deming
selected by default; they do not change the fitted calibrations or guardrails.

The Follow up meeting view also includes Addis seasonal chemistry, with stacked
bars in concentration units or percentages. It uses the existing gallery
chemistry after exclusions, with FTIR EC and OC counted once. Its default is
the Addis local `belg_feb` calendar; a control switches to the frozen meeting's
`dry_feb` bins. Sample windows, counts and per-species coverage accompany the
bars. Percentages are shares of the positive visible species means or medians,
not shares of measured PM2.5. The selected-calibration crossplots show actual
IMPROVE fitting and holdout counts plus held-out R² and RMSE in µg/filter.

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
  and connected scatter together: the 1:1 line (squares the panels and keeps
  equal units per pixel), an axes mode — `Data` (0 always visible, negatives
  kept), `From 0,0` (the default; both axes start at zero; points below are not
  drawn but stay in the fit, and the chart says how many), `Show intercept`
  (keep all observations and mark the selected fits' intercepts, with compact
  margins and independently positioned axes of equal span) — and log axes. Fit
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

- every measurement the filter carries, grouped by family, with a **Reference population** picker: its own site (default), any other gallery site, the four gallery sites pooled, or the available public SPARTAN network. The range, IQR, median, coverage and percentile follow that reference while the selected filter stays fixed. Values outside the range stay visible at the edge and are labelled;
- the ratios worth reading off one filter (HIPS BC/FTIR EC, OC/EC,
  OM/OC, EC/PM2.5) against the selected reference median of the same ratio, using its paired coverage;
- **a mini crossplot with its own axis pickers** — using the same reference population, with this filter ringed even when it falls outside the reference cloud. Its selected regression fits and confidence bands follow the global controls;
- its neighbours in time at the site, click-through, plus ← / → keys to step.
- the baseline-corrected spectrum where available: spline baseline uses its
  4000–1425 cm⁻¹ grid; VIBES uses the full-range 4000–500 cm⁻¹ fit. Selecting
  **both** shares axes while preserving each trace's native grid and endpoint.
  The default VIBES export is the completed rank-70 correction, including
  the two separately recovered held-out blanks. A labelled rank-30 selector
  preserves the earlier correction. A failed full-range load never substitutes the archived truncated fit.
  Use +/− or enable **Zoom / pan** for wheel/pinch zoom and horizontal dragging.
  Absorbance rescales to the visible wavenumber window; **Reset zoom** restores
  the full range and disables gesture capture. A new filter or a change in the
  available spectral range restores the full view.
  **Expand spectrum** opens a larger modal with the same baseline choices,
  full-range traces and zoom controls. Close, Escape or a backdrop click returns
  to the drawer while keeping the filter pinned; keyboard focus returns to the
  Expand button.

Clicking empty plot or page space, closing the drawer, pressing Escape, or
unpinning clears the sample pin, closes its drawer and removes `pin` from the
view URL. Drawer controls and brush selections keep their own interactions.

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
batch_results.jsonl`, 109,924 configuration × site × k rows). No models are
refitted: `export_calibration.py` keeps the rule-k row per configuration ×
evaluation site (14,356 rows) and the full k sweep for Addis under protocol A
(14,092 rows), plus the dated write-ups' first paragraphs. Four charts follow
the argument in those write-ups:

- **Cutoff sweep** — readout vs cohort size, one line per calibration spectra
  space, locked cutoff marked. The 2026-08-20 dense sweep's basin at
  ocec-440–490 × AIRSpec is visible here; the coarse ladder never looked below 600.
  The calibration panel selects held-out IMPROVE R², held-out RMSE, or internal
  RMSECV used during factor selection. Both RMSE measures use TOR loading in
  µg/filter; evaluation-site RMSE is a separate concentration score. The
  horizontal axis defaults to actual filters used in fitting when saved counts
  are available, with requested cutoff retained as an option. A table reports
  selected cohort, fitted and held-out counts at the clicked or locked cutoff.
  Export enrichment reads matching saved fit/CV caches without refitting;
  unavailable historical values remain missing.
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

To recover counts and calibration errors from existing caches:

```bash
uv run aeth doctor
uv run python gallery/data/export_calibration.py --enrich-cache
```

If the external TOR file is still hydrating, use the locally prepared metadata:

```bash
uv run python gallery/data/export_calibration.py --enrich-cache \
  --staged-metadata research/ftir_hips_chem/output/tables/vibes_colab_bundle/stage/data/pool_metadata.csv
```

The staged table supplies candidate filter identities and sites; an exact
resolved-cohort cache hash, matching fitted count, rule factor count and
held-out R² are still required. The adjacent `etad_metadata.csv` also permits
the corrected-space Ethiopia ranking to be reproduced with the shared
selection helper. No PLS model or CV curve is refitted. A gitignored sidecar
in the explorer cache preserves these matches for later ordinary exports.
On 1 October 2026, 14,212 of 14,356 grid rows have counts and internal RMSECV;
10,596 have outer held-out RMSE. Interleaved protocols deliberately have no
outer held-out score. The remaining 144 rows are historical lowest-OC/EC or
raw-space spectral-analog configurations whose exact cache checks failed;
they display unavailable values. For example, the 800-filter spline-baseline
site-grouped configuration fits 606 filters and holds out 194, with outer
RMSE 3.8587 µg/filter and inner RMSECV 6.5587 µg/filter.

One configuration is selected across the page (click a point or a heatmap
row); when it matches an exported preset, the per-filter section below
follows.

The **Specification curve** has dropdowns for cohort selection method,
selection spectra, cohort cutoff, model spectra, cross-validation protocol,
PLS factor count, and calibration lot where available. These filters intersect
with clicked choice strips and the guardrail toggle; the shown/pass counts
refer to the filtered configurations. Reset filters restores the full search.
Multiple clicked choices within the same setting (such as three cohort methods
or both cross-validation protocols) are combined. Choices across different
settings intersect, so selecting cohort methods and a factor range retains
those methods only within that range. Selected choices can also be removed
using the buttons above the plot, including when no saved configuration matches.
The search selector switches between the original spline/raw/derivative export
and the 17 Sep spline/VIBES export. In the latter, selection and model baselines
are paired in the saved runs, so mixed-baseline selections have no results.
Only original-search curve points select configurations in the original
k-sweep and per-filter readouts below.

Graph view switches between one selection at a time (with a selection dropdown),
side-by-side or vertically stacked selection cohorts on a shared vertical
scale, a combined overlay ordered by percentile within each selection, and
the pooled ordered curve with choice strips. The main Calibration tab opens
the spline/VIBES search with those two selections stacked; all baseline
variants are also available. Overlay percentiles compare distributions and do
not pair configurations. Red dots can be hidden independently, or highlight
saved guardrail passers, a configurable near-zero intercept band, slope plus
cross-validation R², or all three custom criteria. Custom criteria omit the
saved OOD/negative-prediction conditions and never alter saved guardrail flags.
The optional matches-only filter follows the active highlight rule.
Choice strips omit the spectral preprocessing block: selection panels and
the baseline controls already identify it. Calibration cohort and settings
rows retain their original saved-mask indices and remain clickable filters.

The VIBES **fit restricted to 4000–1425 cm⁻¹** is archived. Calibration hides
its configurations and dropdown entries unless **Show archived VIBES fit** is
enabled; reset filters retains that archive choice. Counts use the available
active population and separately report hidden archived configurations. The
Follow up meeting view follows the same default. The full-range fit cropped to
4000–1425 for prediction stays available as a clearly labelled sensitivity
comparison, rather than being conflated with the older truncated fit.

The Baselines tab's **One physical filter, both corrections** also supports
overlay, one method at a time, and stacked spectra with a shared vertical scale.
Its range selector offers 4000–1425 or 4000–500 cm⁻¹. The full range adds the
current full-range VIBES correction on its own 8-channel display grid, matched
by physical filter ID; all 12 inspection cases are in that export. The archived
raw spectra, corrections and their difference stop near 1425 cm⁻¹. No values
are extrapolated, and the archived predictions and earlier selection masks
retain their original meaning. Range, layout and method choices persist in
the view URL.

The Baselines tab now opens with the completed rank-70 benchmark, paired
site-bootstrap RMSE intervals, population spectral overlays, held-out blank
residuals, synthetic signal recovery, and current per-filter traces. The
calibration-cohort selector switches the paired source-domain evaluation;
preliminary rank-30 and archived variants can be included explicitly. Spectral
population and range selectors control the median/IQR overlay, whose legend
can hide each correction independently. These results do not establish EC
accuracy in the Addis test set. The run includes all 12,808 finite corrections,
including two held-out blanks recovered separately from the original export.
Seasonal predictions, combined sets, repeat fits and the specification grid
on Follow up meeting use completed rank-70 models by default. The VIBES
analysis selector preserves the earlier rank-30 models as an archive.
**Show archived
benchmark** reveals the original scores, diagnostics and paired inspection
curves without changing their saved data. The default current inspection
uses completed rank-70 full-range VIBES traces by default, with a separate
preliminary rank-30 selector, and omits archived predictions and archived
better/worse labels, and starts on 4000–500 with no selection mask. A shared
window crops only the display of the full-range fit. Spectral similarity also
offers separately recomputed rank-70 full-range and shared-window matrices,
rankings and traces. Its archived truncated VIBES option is hidden by default.
Archive visibility is preserved in the view URL (`carchive`, `barchive`,
`sarchive`; the meeting tab uses `marchive`). Current/earlier seasonal and
specification versions persist as `mversion` and `cversion`.

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

## Regression display controls

The sidebar's **Regression fits** checkboxes select OLS, Deming and York
independently. Deming alone is the default; deselect all to hide fits and
equations. **Reset fits** restores the default. The choice persists across
tabs and in copied URLs (`fits=ols|deming|york`, or `fits=none`). Live crossplots,
scatter matrices, density plots, filter detail panels and calibration readouts
share this selection. Saved coefficient comparisons display one view per
selected estimator; historical exports without the requested coefficients
show an explicit unavailable message.

York uses the uncorrelated-error iteration already implemented in
`calibration_explorer/hips_lab.py`. Live chemistry plots use relative per-filter
HIPS uncertainty when all paired filters carry it, normalized to its median;
the other axis uses a common error scale based on the chart's Deming ratio.
Otherwise both errors are constant. Saved summary views also use constant
errors with their saved Deming ratio, so their York coefficients coincide with
Deming. Captions identify this assumption. These controls change displayed
comparison fits, not PLS calibration models, saved scores or Deming guardrails.

Run `npm test` from `gallery/app` for URL-selection and numerical checks,
including parity with the existing Python York implementation.

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

The **95% fit uncertainty** checkbox is off by default and persists across tabs
and shared URLs (`ci=1`). It shows pointwise line confidence bands and marginal
slope/intercept intervals for each selected estimator. Live plots use 500
deterministic bootstrap resamples of the displayed finite pairs, preserving
x/y/uncertainty alignment and joint slope/intercept covariance. Where all paired
filters have site/date metadata, whole calendar months are resampled within each
site; otherwise the readout identifies a paired bootstrap that assumes independent
pairs. At least eight pairs, six months in each site for block resampling, and
90% valid fits are required. Hover the stats box for the resampling assumptions.
These are exploratory intervals conditional on the displayed PLS model, exclusions,
Deming ratio and York error assumptions; they do not include calibration-training
uncertainty, MAC uncertainty, or individual measurement/prediction errors.

Saved seasonal coefficient intervals follow the toggle; saved weighted York
standard errors become explicitly labelled approximate 95% normal intervals
(±1.96 SE). Cross-site intervals are calculated only when the raw pairs reproduce
the saved estimator coefficients. Saved specification/sweep summaries without
intervals remain point estimates with an availability notice. Separate 10–90%
variability ranges across cross-validation splits and spectral percentile envelopes
retain their own meaning and are unaffected by the 95% fit control. Reset fits
restores Deming alone with fit uncertainty off.

**Fit grouping** in the Fits sidebar switches between one pooled fit and a
separate fit per site (`fitgroup=site` in shared URLs). It applies to the main
scatterplot, density plot, scatter matrix, sample-drawer crossplot, multi-site
meeting analog crossplots, and the Addis/Bishoftu Baselines plots. The same
styled control is available beside those charts. Per-site fits use only finite
paired values from that site, including aligned uncertainty vectors and month
blocks, and cover that site's observed x range. Sites need at least three pairs
and nonconstant axes. Statistics appear below the plot; meeting analog plots
keep their longer per-site readouts in an expandable section. Site colours come
from gallery metadata where available; meeting point colours retain their
role/season meaning. Matrix correlation summaries remain pooled for the selected
population. Baselines chart controls can explicitly override the gallery setting;
their default follows it, while two-band fits stay hidden in default pooled mode.
Reset fits also restores pooled grouping. Single-site calibration plots and
existing per-site transfer panels already have one fit for each displayed site.

Drawer reference choices persist in copied URLs (`sampleRef` query parameter).
Site/four-site references use the complete gallery export after canonical
exclusion flags, independently of the main graph's subset. The public-network
reference is `data/spartan_reference.json`, generated with
`uv run python gallery/data/export_spartan_reference.py`: 4,363 physical filters
at 37 locally available ChemSpecPM25 sites over all available dates, with known
study exclusions flagged and removed by the reference selector. It uses actual
filter values, not averaged site medians. Carbon is the public, rounded FTIR
product; the EC/OC labels do not represent independent measurements. Public BC
is not substituted for HIPS BC. HIPS and FTIR functional groups have no network
reference here, so their selected-filter values remain visible with an explicit
unavailable comparison. The public reference retains upstream quality decisions
at sites without a study exclusion registry. Ratios require paired fields;
percentiles need at least four finite reference values. Network points lacking a
full gallery record have hover details without opening a nonexistent record.

### Completed rank-70 VIBES migration (2 October 2026)

Current Baselines views include paired residuals, loading strata, exact spectral
region decomposition, blank distributions, common-test calibration selection,
spectrum-only routing, training-only PCA and the 31-spectrum external pilot.
The Addis/Bishoftu optical and two-band scatterplots offer no fitted lines,
one pooled fit, or separate fits per site. Site fits match the point colours,
use only that site's paired values, and cover its observed x range. Equations,
R² and pair counts appear beside the plot; the Fits sidebar controls estimators
and uncertainty. Optical Deming fits retain the configured HIPS error ratio;
two-band Deming fits assume equal error variance. These display fits do not
refit the frozen EC prediction models.
Seasonal/combined calibrations were refitted on both the full (2,722 channels)
and shared (2,002 channels) corrected grids. The category grid is complete:
9,010 fits, including 3,138 unchanged spline fits and 2,936 per VIBES window.
Original calibration panels below the specification curve use spline/raw/derivative
inputs; they have no VIBES results to migrate. Existing SPARTAN chemistry products
remain measured/source products and are not replaced with these experimental models.

Rebuild from the validated recovered correction with these commands, in order:

```bash
uv run aeth doctor
uv run python research/ftir_hips_chem/workflows/migrate_vibes_gallery_rank70.py meeting
uv run python research/ftir_hips_chem/workflows/migrate_vibes_gallery_rank70.py grid
uv run python research/ftir_hips_chem/workflows/migrate_vibes_gallery_rank70.py export-meeting
uv run python research/ftir_hips_chem/workflows/migrate_vibes_gallery_rank70.py followup
uv run python research/ftir_hips_chem/workflows/migrate_vibes_gallery_rank70.py export-followup
uv run python research/ftir_hips_chem/workflows/export_vibes_rank70_paired.py
uv run python gallery/data/export_baseline_applicability.py --completed
uv run python gallery/data/export_similarity.py --completed
uv run python research/ftir_hips_chem/workflows/migrate_vibes_external_rank70.py
```

The grid and external corrections resume only against identical input hashes.
The meeting export rejects incomplete or duplicated grids. Derived outputs live
in `output/tables/vibes_fullrange/gallery_rank70_20261002`; generated gallery
assets are git-ignored. See [migration audit](../docs/vibes-gallery-rank70-migration.md)
for source provenance and coverage.

### Native SPARTAN network data and loading

A preview workspace without the native source bundles can restore the existing
published export with `uv run python gallery/data/restore_spartan_assets.py`.
The command verifies every asset against its content-addressed filename and
records source/asset hashes in `spartan_network/restoration.json`. It copies
existing measurements and corrections without refitting. Include the complete
`spartan_network/` directory when packaging a Site; its index alone is not enough.

The chart site picker uses `public/data/spartan_network/index.json`, a small
site/coverage directory. Only the selected sites' compressed filter tables load.
Opening a filter loads that site's native record directory and that filter's
spectrum file, independently of its chemistry. All spectra stay at native
resolution; repeat analyses are selectable. The drawer's **FTIR spectrum** block
switches between Raw, Spline (AIRSpec), VIBES and Both; the corrections are
precomputed for every analysis (below), never fitted in the browser. For Addis
filters, “Calibration-run spectrum” (collapsed) still loads the frozen runs'
corrected traces, 10 cm⁻¹ bins, the ones the calibration used.

**Stored traces** (`gallery/data/spectrum_codec.py`, decoded by `lib/spartan.ts`):
integer steps of 1e-8 absorbance, stored as differences from the previous point,
with each 32-bit difference split into four byte planes, then gzip. The rounding
(≤ 5e-9) is below float32's own spacing, and a file is about half the size of
float32: measured on 400 spectra, 10.6 KB → 5.5 KB per filter (1e-7 steps would give
4.3 KB with rounding still ~40× below the point noise; `--step 1e-7`). Gaps are kept
as runs. Each analysis carries `series.raw`, and where the baseline run has it,
`series.airspec_baseline` (native grid, 1425–4000 cm⁻¹; corrected = raw − baseline)
and `series.vibes_corrected` (on the VIBES background's 2,722-point grid, given in
`index.json` `baselines.vibes.grid`).

**Baselines** come from `research/ftir_hips_chem/workflows/network_baselines_20261019.py`:
AIRSpec (APRLssb spline, df 6/4) on each native grid, locally (~2 min for all
4,452 analyses); VIBES with the gallery's saved background (87 blanks, rank 70; no
refit), after averaging each native spectrum onto that background's grid, on a
Colab VM. The exporter picks them up from that run's folder when present and
checks each against its spectrum's length. Non-Addis VIBES traces use a background
learned from IMPROVE and Addis blanks, which the drawer says.

Configure the unpacked source folders through `AETHMODULAR_SPARTAN_FTIR_BUNDLE`
and `AETHMODULAR_SPARTAN_HIPS_BUNDLE` in the gitignored repo `.env`, then run:

```sh
uv run aeth doctor
# Rebuild the existing public/lab chemistry first if those sources changed:
uv run python gallery/data/export_spartan_reference.py
uv run python gallery/data/export_spartan_network.py
```

The exporter verifies both native bundles' source hashes, keeps the native
archives outside the gallery, derives `−log10(sample/background)` with each
analysis's own scans and scaling, and writes gzip assets with content hashes
in their names. Nonpositive single-beam pairs become gaps; unsupported formats,
missing scan references, and background extrapolation stop the export. The
index is replaced only after its assets are complete. Native FTIR and HIPS
coverage is reported separately from public and delivered measurements. Raw
HIPS transmittance/reflectance is never substituted for delivered Fabs or BC.

The 19 October 2026 snapshot has 40 known site entries (38 with filter rows),
5,182 physical PM2.5 filters across the existing chemistry and native metadata,
and 4,452 native FTIR analyses across 28 sites. The native spectrum archive also
retains PM10 and field blanks with their types; chart rows and the PM2.5 drawers
never substitute those for PM2.5 measurements. It includes 4,274 registered-site
HIPS sample readings. Unassigned native lab blanks stay in the original bundle.
Missing native records are coverage gaps; public measurements are retained.
Filter counts are distinct from non-null measurement or matched-pair counts.

Measured files for this snapshot: the default four sites' filter payloads total
about 173 KB compressed. The previous monolithic network reference is 4.44 MB
as JSON, or about 701 KB gzipped: the default site transfer is about 75% smaller
when comparing compressed sizes. All sites' filter payloads total about 730 KB; a filter's spectrum
file (raw plus AIRSpec) is about 7 KB compressed, 32 MB for the network. These are data-transfer sizes, not a
page load-time benchmark. Calibration exports load separately after the initial
filter view and only on pages that use them.

Both full publication and `gallery/share/build_share.py` retain the site and
spectrum shards, so online site picks and drawers have the same coverage as
local ones. The loader decompresses explicit `.gz` files in the browser; static
hosting needs no special `Content-Encoding` setup. The publication scanner
inspects compressed JSON as well as ordinary text. Regenerated shards are
ignored by Git and travel with the existing gallery data snapshot, rather than
in the source repository.

### Checking what a network update adds

```sh
uv run python gallery/data/audit_spartan_update.py
```

This compares physical IDs against the union of the old focus and network
exports, FTIR analysis IDs against the previously staged site pulls and Addis
spectra, raw HIPS readings against the previous archive, and downloaded public
chemistry files against the local mirror. Summary JSON and per-ID/site CSVs go
to `research/ftir_hips_chem/output/tables/spartan_network_update/`; hashes pin the
compared inputs. Override the previous-source options when comparing another
snapshot. Counts retain native repeats and identify their filter populations.

The October 19 bundle adds 81 physical PM2.5 filters beyond both old gallery
exports, 3,383 FTIR analyses beyond the previous 1,069 (23 additional sites),
and 152 raw HIPS sample analyses / 304 sample-and-reference readings. No previous
analysis IDs or raw HIPS readings were lost, and common HIPS measurement values
were unchanged. All 74 downloaded public chemistry files match the local mirror.

When combining focus filters with network sources, the app now fills both absent
fields and explicit nulls. This recovers 96 EC values, 96 OC values, and 51 HIPS
Fabs values on existing focus filters, alongside other chemistry. These are
previously unshown available products, not new independent carbon measurements;
ChemSpec carbon remains a duplicate FTIR product. Existing non-null values,
including zero and the more precise in-house FTIR values, retain precedence.
Exclusions from either source remain flagged. The drawer lists fields filled
from the network export. Frozen calibration predictions are not rerun.

The Pages workflow uses one concurrency group with cancellation enabled: a new
deployment supersedes older ones. Run names include the source commit and data
snapshot so the current version can be identified in Actions.
