# Gallery chart standards

Audit of every chart in `gallery/app` (1 Oct 2026): 22 charts on the gallery
tabs, 18 calibration charts, and about 30 charts drawn inline in the research
pages (Meeting follow-up, Similarity, Baselines, Census). The question was
which features each chart re-implements instead of sharing, so that each
feature has **one** definition.

The short version: the frame, controls, axes and tooltip are already shared
and mostly used. What is redefined chart by chart is everything *inside* the
plot: reference lines, domains, colours, number formats, legends, selected and
pinned marks, and three whole chart types (crossplot, spectrum, dot-interval
rows) that exist in 6–8 copies each.

Paths below are relative to `gallery/app/src`.

For new work the rule is in AGENTS.md ("Gallery charts: extend the shared
pieces, never redefine them"): use the piece listed here, and when it falls
short, extend it rather than writing a local version.

---

## 1. Standards, one per feature

Each row is the rule we standardise on, the primitive that carries it, and how
many places currently deviate. **New** marks a primitive that does not exist yet.

| Feature | Standard | Primitive | Deviations |
|---|---|---|---|
| Panel, title, explain, export | Every chart sits in `ChartFrame` | `components/ChartFrame` | 0; export composes the whole panel (§4b Export) |
| Controls | `Select` / `Segmented` / `Toggle`, never raw `<select>` or checkbox | `components/Controls` (re-exported by `components/ChartFrame`, so old imports work) | `DynamicSeasonEditor` keeps raw `<select>`s in its condition rows |
| Sizing | `useDimensions`; square crossplots ≤ 560 px, min 240 | `hooks/useGalleryData`, `components/FacetGrid` | Hexbin 520; Ext pilot fixed viewBox ×3 |
| Margins | `MARGIN` unless the chart needs a label gutter; a gutter is measured from its labels, never characters × px | `lib/theme`, `lib/fitLabel` (`labelGutter`, `fitsWidth`, `clipText`) | Literal margins remain in older charts |
| Axes, ticks, titles | `XAxis` / `YAxis` (11 px ticks, 12/600 titles); band scales for category rows; vertical grid lines through `XAxis` `gridHeight` | `components/Axes` | Hand-drawn axes remain where `XAxis`/`YAxis` lack a feature (see Follow-ups in §4b) |
| Domain | `pairDomain` for crossplots; `robustDomain(vals, lo, hi)` returning clamp + pinned count for clipped charts; shared ranges across panels from the panel group (`share`: x, y and a right-hand `y2`), never computed by hand | `lib/axes`, `components/PanelGroup` | Each chart keeps its own quantiles (§4b) |
| 1:1, zero, one, marker lines | One style per kind (identity `5 4` w1.5 `INK.identity`; zero/one solid `INK.axis` .7; marker dashed `3 3` + mono 10 label) | `components/RefLines` (`RefLine`, `IdentityLine`, `Marker`, `RefBand`) | 0 (§4b) |
| Fit lines and equations | `RegressionLines` for every fit, saved or live; `equation()` for text (3 dp, "y =") | `components/RegressionOverlay`, `lib/regressionPrefs` | CrossSiteTransfer clips by hand; 3 equation formats (`fmtFit` 2 dp, `equation` 3 dp, inline strings) |
| Fit summary placement | Above the axes at every width: n / R², then one equation per line; group summaries stack on phones and use up to two columns on wider panels. Wrap long text at a fixed readable size and retain it in exports. | `components/RegressionOverlay.useRegressionSummary` | Applied to scatter, hexbin, shared research crossplots, station comparisons, and filter drawer crossplots. Specialised saved-result legends retain their own layout. |
| Intervals | Whisker bars share one style | **new** `IntervalBar` (beside `RegressionBand`) | HipsYork, CrossSiteTransfer, Meeting ×2, BC LoadingPlot: 3 whisker styles |
| Guardrails | Slope box and held-out floor drawn one way | `components/RefLines` `RefBand` | 0 (CutoffSweep, KSweep, SlopeInterceptTrap, SpectraDumbbell) |
| Tooltip | `useTooltip` only; no native `<title>` alongside it. A mark that is not a filter point (a bar, a cell, a band) spreads `tip.bind(body)` | `hooks/useTooltip` | `lib/tooltips.test.mjs` fails on a file with both; its `ALLOWED` list keeps `<title>`s that only restore a clipped label |
| Tooltip content | `id`, `site · date`, `name: value unit`, trailing "click for details" | `components/ScatterLayer` `tipOf` (the hover, tooltip and open handlers are its defaults) | The copies of hover + tip + open are gone (29 Oct); the 4 wordings (`x: v`, `x = v`, `≈`, `median =`) remain |
| Legend | `Legend` / `useLegend` as a centred HTML row inside `.chart-wrap`; grouped site/season keys centre as one aligned block; `Key` for a static key (`Legend` without toggles); `ColorLegend` for every ramp. Explicit side legends stay in their column. A chart hands `Legend` its items in drawing order; a reader's own order (export dialog, Legend order) is `Legend`'s sorted copy, never the chart's array, so it moves no mark. | `components/Legend` (shapes; `useLegend` `background` keys, `hideBackground`, `external`), `components/Key`, `ColorLegend`, shared styles | The Baselines in-SVG legends are retired; whole-panel export keeps HTML legends |
| Highlight, pin, dim | `focusStyle` for a filter mark's look; `FocusLayer` draws the focused filter over its chart, `PinRing` marks a pin; **new** `selectedStyle` for a selected configuration | `lib/highlight` | 6 selected-config styles |
| Points (filters as marks) | One `ScatterLayer` per plot area: memoised marks, `xy`, `idOf`, `focus`, `tipOf`; one pointer layer picks the nearest point. No per-mark handlers, no reordering on hover, no `useHighlight()` without keys. What sits behind the points (a binned density's cells) answers a hover through the same layer's `onMiss` (pointer over the plot, no point near; `null` when it leaves), not handlers of its own | `components/ScatterLayer`, `components/HitLayer`, `lib/highlight` | 0 (29 Oct); `lib/pointLayers.test.mjs` fails on a new copy |
| Brush | One accent brush style | `lib/brush` (`BRUSH_STYLE`, `styleBrush`) | 0 (§4b) |
| Season shading | One component for bands / strips / arcs | `components/SeasonBands` | 0 (§4b) |
| Colours | Site and season colours through `lib/siteColours` (never `meta.sites.find(...)?.color`); INK tokens; named colour maps in `lib/theme` (`BASELINE_COLOR`, `PREPROCESSING_COLOR`, `LOT_COLOR`, `AETH_CHANNEL_COLOR`, `VARIABLE_COLOR`, `SPECTRAL_REGION`); categorical series from `lib/palettes` (`TABLEAU`, `RESEARCH_SERIES`), never a Tableau hex typed in a chart | `lib/siteColours`, `lib/theme`, `lib/palettes`, `lib/labels` | INK values still typed as hex in some older charts |
| Heat maps | Every matrix is a `HeatMatrix` (measured gutters, readable cell ink, ramp legend, one pointer handler); site × month grids are `SiteMonthMatrix` | `components/HeatMatrix`, `components/SiteMonthMatrix`, `components/HeatCell`, `lib/heatMatrix` | Correlogram scatter matrix, FlagStudy background grid, MinuteCharts `Heat2d` |
| Numbers | `fmt` (null → "—"), `signed`, `pct(v, digits)`, `percent`, `trim`, `adaptive`, `fitSummary`, `fitLine`; one unit prettifier | `lib/format`, `lib/theme` (`withUnit`, `prettyUnit`) | Some `toFixed` calls remain in older charts |
| Statistics | Every statistic from one module, null-tolerant, with `minN` where a minimum count applies | `lib/stats` (`monthlyStats`, `corrMatrix`, `pearson`/`spearman`, `boxStats`, …) | 0: `sections/research/common/stats` is gone |
| Data loading | One fetch + schema check + loading/error panel | `lib/data` (`useExport`, `useLoaded`, `useDerived`, `useEach`), `components/ExportState` (`LoadState`) | 0 `let live` effects outside `lib/data` |
| Empty state | `Empty`, and the chart is not drawn behind it | `components/ChartFrame` | AvailabilityHeatmap none; Treemap and BubbleMap draw an empty chart under it |
| Transitions | `.animated` class | `styles.css` | 5 charts snap; 3 use inline transitions |
| View state | `useSearchParam` (or `useChartOptions`) for anything a link should reopen | `lib/url`, `lib/chartOptions` | `useState` controls remain on some research pages |
| Height in a tile | The chart's height (or row height, or square side) from `useFitHeight() ?? its own`; legends through `Legend` so a tile can compact, move or hide them | `lib/chartFit`, `components/Legend` | Wind roses (fixed-size roses) and multi-panel layouts (per-site grids) keep their size and scroll inside the tile |
| Dashboard tile | One `ChartFrame` per chart; registered in the Add chart menu; settings in `useChartOptions` / `useSearchParam` so a shared dashboard reopens them | `components/dashboard/registry`, `lib/url` | The Weather and Calibration charts join through adapters (`components/dashboard/extraTiles`) |
| Small constants | `MONTHS`, `MONTH_LETTERS`, heat ramps defined once | `lib/theme` | all month labels, including `lib/heatMatrix` |
| Floating layers (popovers, notes, menus) | Portalled to `<body>`, `position: fixed`, placed from the trigger and kept inside the window; never `position: absolute` inside a panel, where a scrolling box, a tile or a phone's edge cuts it off | `hooks/useFloatingMenu` (menus, pickers), `ChartFrame` `InfoTip` / `useFloatingTip` (notes), `hooks/useTooltip` (hover readouts) | Header-only popovers (Views, nav menus, Dashboard header panels) still sit in the header; `src/lib/floating.test.mjs` fails on any new in-place popover |
| Excluded filters (Filters → excluded) | Hide (default) / Include (`excl=1`) / Mark (`excl=mark`). Under Mark a chart that draws filters as points draws the excluded ones as a muted × (never in a fit, R², median, count or axis range; the hover says why); a summary chart says how many it leaves out. One rule decides what counts (`useCountedFilter`): a filter is out when excluded by hand or outside `keep=`, or when it is one of the export's exclusions and the mode is not Include; every point, fit, season or group line, median, bootstrap, table and count built from per-filter data uses it (Crossplot applies it to its points and refit itself; anything a page computes from the same items goes through `useExclusionSplit`). Only a chart that draws saved statistics no browser refit can redo says the switch does not act there (`FixedSetNote`), and a Crossplot line read from an export is flagged `saved`. A page with its own per-filter items (explorer predictions, holdouts, spectra, scores; MediaId tables through `lib/etadMedia`) joins them to the exclusions by filter id; the sample drawer's reference population follows the same rule; a meeting page As presented counts every filter it was shown with | `lib/exclusionMode` (modes, `splitById`, tested), `lib/markedRows` (`useCountedFilter`, `useMarkedRows`/`useMarkedPoints`, `useExclusionSplit`, `useKnownFilter`, `ExcludedNote` with `count`, `FixedSetNote` with `hand`/`subject`, `toMarks`); drawing through `ScatterLayer` `marks`, `Crossplot` `marked`, `SweepPlot` `marks`, `Figure` `markedData` and `aside` | 0 (30 Oct): every page that draws filters follows it; no page keeps its own exclusion check |
| Stepping (previous / next) | One chevron button, square and as tall as the `Combobox` beside it; no text arrows, no taller touch buttons beside a picker | `components/StepButton` | 0 (single-chart view, Connected scatter) |
| Swap axes | A small curved two-way icon without a visible box in the lower-left margin, operable by click, Enter or Space and omitted from exports. Editable pairs use their field provider; fixed comparisons retain their own orientation per plot/tile (`axes.swap.<key>` in the URL; panels that swap together share a key). Live fits refit with the reciprocal error-variance ratio; saved lines, bands and backgrounds transpose with the points. | `components/Axes` (`AxisSwap`, `useAxisSwap`, `OrientedPlotLayer`), `lib/plotOrientation` (tested); built into `Crossplot` and `ScatterPlot` (`swap`; on by default in the experimental `Scatter`), and the hand-drawn crossplots that predate them | The scatter matrix already displays both orientations of each pair. |
| Network rank options | One site-median rank plot for IMPROVE, SPARTAN or both: comparison networks, notebook placement in the IMPROVE ordering, SPARTAN scope, background colours, population rule, minimum counts, linear/log; active Samples sites are stars in their own colours that replace their background dot; the quantity comes from the shared axis picker (notebook ratios as presets). | `charts/ranking/NetworkRankPlacement`, `lib/referenceRank` (`rankNetworks`, `rankWithActiveSites`, `NETWORK_RANK_PRESETS`; tested); [gallery-network-rank-options.md](gallery-network-rank-options.md) | 0 (31 Oct) |

## 2. Composite charts that exist in many copies

These carry most of the duplication. Each should become one component that
uses the primitives above.

1. **Crossplot**: Scatterplot, DensityHexbin, ConnectedScatter, RunCrossplot,
   AnalogLab, CrossSiteTransfer multiples, Meeting `Scatter`, Ext
   `AllFilterScatter`. They share square sizing, `pairDomain`, origin-crossing
   axes, the identity line, intercept markers, fit lines and the "points
   dropped" note. Today: 3 identity styles, 2 sizes, 3 scale builders, and the
   Meeting and Ext copies ignore the axes policy (they keep negative points
   while the Correlation tab's default "From 0,0" hides them).
2. **Spectrum plot**: FilterSpectrum, `SpectraOverlay`, Meeting `panel` and
   `SpectraPanel`, Similarity `SpectraTraces`, Baselines `SpectraPlot`. They
   differ in y-domain, grid, clip-path id (two fixed ids would collide),
   tooltip, legend and excluded-region shading (3 colours). Trace loading is
   also implemented 3 times (`lib/filterSpectrum`, Meeting `useTraces`,
   Similarity `useBinary`).
3. **Dot-interval rows** (forest plots and lollipops): CorrelationRanking,
   Similarity `GroupRanking`, Meeting `RepeatStrip` and `Combined`, Baselines
   `LoadingPlot`, HipsYork, CrossSiteTransfer ordering. 3 whisker styles, label
   gutters of 130/250/260 px.

## 3. Bugs and visible contradictions to fix first

These are small and independent of the refactor. **Done 1 Oct**: ConnectedScatter
clip, `.data-table` (now `census-table`), duplicate native tooltips (map + 5
Baselines charts), null cells and the Ext `fmt` guard, SeasonalClock IQR now
n≥4, and the Meeting keys and text now follow the active estimator. Still
open: export (step 4) and the colour clashes (step 2).

- `charts/correlation/ConnectedScatter.tsx`: no clip path or `inDomain`, so in
  "From 0,0" mode a negative median draws into the margin.
- `ChartFrame` export (`components/ChartFrame.tsx:46`) serialises only the
  first `<svg>`: faceted charts export one panel, and HTML legends are dropped.
- Meeting follow-up text says "black line", but the fit is drawn in the
  estimator colour (magenta dashed for Deming); the ThreePlots key is fixed to
  Deming even in the OLS and York views.
- `.data-table` (CutoffSweep) has no CSS rule, so the table is unstyled.
- `SiteBubbleMap` and 5 research charts show the native browser tooltip on top
  of the custom one.
- `SlopeInterceptTrap` table cells render empty for null (missing `?? '—'`);
  `BaselineExternalPilot` local `fmt` has no null guard.
- MonthlyBand says "IQR n/a (n<4)" for a site-month that SeasonalClock draws
  an IQR for (n≥3), although its comment says they match.
- Same baseline, different colour by tab: AIRSpec is `#2C6E9E` on Baselines and
  in the drawer, `#2171b5` (the accent and OLS colour) on Meeting and
  Calibration; full-range VIBES is purple, orange or red depending on the page.
- One colour, several meanings: `#c026d3` is the Deming line, 2nd-derivative
  spectra, a HipsYork variant and a SelectionRanking series; white-filled
  markers mean "selected", "rule choice" or "pinned off-axis" depending on
  the chart.

## 4. Proposed order

1. **Bugs** (§3). Small, no design decisions.
2. **Tokens and formatters**: `METHOD_COLOR`, region shade, INK hex
   replacements, `fmt` / `fmtSigned` / `fmtPct`, one unit prettifier, `MONTHS`,
   `HEAT`. Mechanical, mostly invisible except the colour fixes.
3. **Primitives**: `RefLine` / `IdentityLine` / `Marker`, `XAxis gridHeight`,
   band-scale row labels, `robustDomain`, `selectedStyle`, `IntervalBar`,
   `GuardrailBands`, `filterMarkHandlers`, `useD3Brush`, `SeasonBands`,
   `useLegend` background keys, `HeatCell`, `useExport`, `useFacetGrid`.
4. **Composite charts**: `Crossplot`, `SpectrumPlot`, `DotIntervalRows`, and
   whole-panel export (all facets plus legend in one SVG).

Each step should leave the published numbers unchanged; where a chart's
picture changes (colours, decimals, domains), the change is listed in that
step's commit.

## 4b. Status (14 Oct 2026, updated 31 Oct)

Done, and each defined once in `gallery/app/src`:

| Feature | Lives in | Notes |
|---|---|---|
| Baseline colours (`BASELINE_COLOR`, `baselineColor(method, rank)`), IMPROVE grey, Addis-all yellow, bank colours, spectral regions, pinned/hovered trace style | `lib/theme.ts` | The Baselines exports call every VIBES run "VIBES"; `lib/baselineRank.tsx` gives each view its run's rank so rank-70 VIBES is orange on every tab. `#2171b5` is now only accent and OLS. |
| Baseline names, short names, EC axis titles | `lib/labels.ts` (`baselineName`, `PREPROCESSING_SHORT`, `ecAxis`) | |
| Number formats | `lib/format.ts` (`fmt`, `f2`, `f3`, `signed`, `pct(v, digits)` for a fraction, `percent` for a value already in percent, `trim` for ticks, `adaptive` for ramp legends, `sci`, `fitLine`, `fitSummary`, `coef`, `coefSigned`) | `lib/stats.fmt` re-exports it. The §5 precision is applied (11 Oct): slopes, intercepts and R² at 3 dp through `coef`/`coefSigned`, `FitCell`/`BandCell` and `fmtFit`; concentrations stay at 2 dp. A fit's one-line text is `fitSummary`, not a template string per chart. |
| Seasons | `lib/seasons.ts` | Two Addis calendars on purpose (docs/site-seasonality.md): pages showing frozen calibrations use `dry_feb` and say so (`ANALYSIS_CALENDAR_NOTE`); names are normalised to one dash. |
| Data loading | `lib/data.ts` (`loadJson`, `loadBinary`, `useExport`; `useLoaded(key, load)` for any other promise, `useDerived` for a result built from loaded files, `useEach(codes, load)` for one file per site), `lib/calibrationFiles.ts` (`useCalibrationFile`, `useCalibrationRuns`), `components/ExportState` (`LoadState` for the loading/error line, `ExportState` for a whole panel) | One cache: every file downloads once per session. Spectra: `lib/similarity.loadTraces` checks the grid against the filter table for every page and the drawer. A page or chart never writes its own `useEffect` with a `let live` flag: the hooks above settle a stale load for it. |
| Pickers | `components/Pickers.tsx` (`BaselinePicker`, `SeasonPicker`, `BankPicker`) | Page option lists come from data or context, not module `let`. Raw `<select>`/checkbox controls replaced. |
| URL keys | `lib/url.ts` | Development warning when two controls use one key with different options; the `sregion` collision is fixed. |
| Points: hover, pin, click, right-click (29 Oct 2026) | `components/ScatterLayer.tsx` over `components/HitLayer.tsx` and `lib/highlight` (`FocusLayer`, `PinRing`) | Crossplot, Scatterplot, DistributionPanel, Timeseries, AERONET timeline / typing / scale height, Weather timeline, DailyTimeline, RainWashout, RunSeries, RunCrossplot residuals, CrossSiteTransfer, PwMoisture, the Experimental `Scatter`, Satoshi overlap. The scatter matrix uses the same pieces on an overlay (up to ~140,000 dots). Not points, so not on it: ConnectedScatter (medians), DensityHexbin (bins), AnalogLab (anonymous pool), SpectrumPlot, ParallelCoordinates, GroupDrawer. A hover stays in its card (`ChartFrame`): it lights the filter up on that card's charts only, and a chart alone in its card touches nothing else; a pin holds on every chart. A chart reading the hover from a hook above its own frame passes the card id to both (`useHighlight(keys, hoverCard)`, `<ChartFrame hoverCard={hoverCard}>`). Guarded by `lib/pointLayers.test.mjs`. |
| Crossplot | `components/Crossplot.tsx` | Meeting, Satoshi (`Cross`, `Envelope` with a density `background`), `RunCrossplot` (residual panel kept beside it on the same x), Ext `AllFilterScatter` optical panel (its own `grouping`). The Ext band-height panel is not a 1:1 comparison and keeps its own axes. |
| Legend and static key | `components/Legend.tsx` (`Legend`, `useLegend`), `components/Key.tsx` | `Key` is `Legend` without `onToggle`/`onHover`, so it gets the tile modes, export colour tags and the shared swatch shapes (dot, ring, square, band, line, dashed, diamond, hollow diamond, bar). `useLegend(hidden, { background, hideBackground, external })`: `background` keys name shading behind the data (seasons), hover deepens it and dims no series, a click hides one only with `hideBackground`; `external` entries mirror an overlay toggle beside the chart. The Baselines in-SVG legends are retired: whole-panel export keeps the HTML legend. |
| Spectrum shading | `components/SpectralRegions.tsx` | Meeting, Satoshi VIP, Blanks. The Baselines amber band marks the earlier selection masks (a different meaning) and stays. |
| Dot-interval rows | `components/IntervalStrip.tsx` | Meeting `RepeatStrip`, `Combined`; Satoshi `Swap`, `BlankEffect`; Baselines `LoadingPlot`; `HipsYork`; Similarity `GroupRanking` (stems, selectable rows, labelled marker). `CrossSiteTransfer` ordering is a jittered dot plot with per-dot picking, a different chart, and stays. |
| Range rows | `components/RangeRows.tsx` | Rows with their own axis each (linear or log): a reference band (outer range, inner range, median), dashed reference ticks, points under test and a verdict column. Interactive legend (hide rescales each axis, hover dims), gallery tooltips, label gutters measured from the text, row height from the fitted height. First user: Adama TOR study `PilotPosition` (28 Oct 2026). Legend swatches gained a `diamond` shape for it, drawn as a diamond in exports. |
| Rolling statistics | `lib/rolling.ts` (`rollingByTime`, `rollingBy`, `rollLabel`, node-tested against pandas), `components/RollingControls.tsx` (`useRolling(chartId, defaults)`) | A rolling line over dated points: window in days (preset buttons or any typed number), median or mean, trailing (pandas `rolling('ND')`) or centred, a minimum count per window; all kept in the link as `<id>.roll` / `.rollstat` / `.rollalign`, with Reset back to the chart's default. Default is ftir_29's trailing 45-day median with at least 5 filters. Users: calibration Dated series (28 Oct 2026), which had drawn a centred ±45-day window with no minimum, and Aethalometer `AgreementOverTime` (`useRolling` with `rollingBy` for its own statistic). Still its own: `aae.rollingMean` (count-based, a different statistic). |
| Heat maps | `components/HeatMatrix.tsx` (any matrix), `components/SiteMonthMatrix.tsx` (site × month), `components/HeatCell.tsx`, `lib/heatMatrix.ts` (the arithmetic, node-tested), `lib/theme.inkOn` | Row and column gutters measured from the labels, cell values printed in readable ink where they fit, the ramp under it (`ColorLegend`, also the palette picker), a tooltip per cell and one pointer handler for all of them, so a hover re-renders no cells. Users include CrossSiteHeatmap, OverlapMatrix, the Similarity matrices, Completeness, AvailabilityHeatmap, MonthlyHeatmap, AeronetClimatology, the weather summaries and the meeting pages; `Figure` `heatmap` draws one from a table. Not on it, on purpose: the Correlogram (its cells, drawn with `HeatCell`, share a layout with the scatter matrix), the FlagStudy background grid (also `HeatCell`), MinuteCharts `Heat2d` (a 2-D histogram on numeric axes). |
| Table cells | `components/TableCells.tsx` (`FitCell`, `BandCell`) | Meeting and Satoshi fit and median-interval cells. |
| Reference lines | `components/RefLines.tsx` (`IdentityLine`, `RefLine`, `Marker`) | 1:1 lines, slope-1 / intercept-0 / zero lines and labelled markers on every calibration, correlation, Meeting, Satoshi, Baselines and Blanks chart (39 hand-drawn lines). Crosshairs, the time cursor, stems and fit lines are not reference lines and keep their own styles. |
| Clipped domains | `lib/axes.ts` (`robustDomain(values, lo, hi, { include, limit, fallback })`, returns the domain and the counts above and below) | SpectraDumbbell, SlopeInterceptTrap, CutoffSweep, SelectionRanking, SpecCurvePlots, Scatterplot, Correlogram, Ridgeline, Histogram, ParallelCoordinates, MonthlyHeatmap, Meeting slope-intercept map, Satoshi ladder and envelope, Blanks raw spectra. Each keeps its own quantiles; they now share one implementation and one pinned count. |
| Season shading | `components/SeasonBands.tsx` (`SEASON_SHADE`, `seasonShade`, `timeSeasonBands`, `monthSeasonBands`, `<SeasonBands>`) | Timeseries, MonthlyBand, SourceStack (strip), SeasonalClock arcs, MonthlyHeatmap strips. One set of opacities, including the legend-hover deepen/fade. |
| Brush | `lib/brush.ts` (`BRUSH_STYLE`, `styleBrush`) | Scatterplot, ParallelCoordinates, DateBrush: accent fill .14, handles .6, no transition. A style helper rather than a hook, because each chart owns its d3 brush lifecycle. |
| Export | `lib/export.ts` (`composeFigure`, `layoutFigure`, `renderFigure`), `components/ExportDialog.tsx`, `lib/exportColors.ts` | PNG, JPEG, SVG and PDF use the same composed figure and color overrides, including every facet, legend and readout. Site colors can be individual or shared; season colors can be separate by site or consolidated by matching names and months; estimator overrides also color equations and confidence bands. Mark other HTML text with `data-export="text"` to include it. Legend order (`components/ExportLegendOrder`, `lib/legendOrder`) is kept per chart in this browser: `Legend` draws it for the file while the dialog is open, and on the page only if the reader asks, so every placement lays out the new order itself. |
| Spectrum chart | `components/SpectrumPlot.tsx` (helpers `lib/spectra.ts`) | Meeting `SpectraPanel` and `BaselineCompare`; Blanks raw spectra, variability, PCA inset, per-blank error inset, components, after correction; Satoshi `Vip` and fingerprint panel; calibration `SpectraOverlay`; full-range VIBES aggregate spectra. Per-series wavenumber grids, one hover mechanism (`nearest` names a trace and pins it, `readout` lists every line), pin linked to the page with `highlight={useHighlight()}`, `markers`, `background`/`children` render props. Left on their own on purpose: `FilterSpectrum` (zoom and pan), `SimilaritySpectra` (shading from a channel mask), Baselines `SpectraPlot` (difference and hidden-mask modes). |
| Sweep chart | `components/SweepPlot.tsx` | Calibration `CVCurve`, `KSweep` (three facets); Blanks rank selection and capture; Satoshi ladder (point scale with a divider) and blank-bank rank CV; Meeting MAC envelope (log x); full-range VIBES injection recovery. Bands or whiskers, the agreement value, labelled markers, the chosen setting ringed, clipped points hollow at the edge. Left on their own: `CutoffSweep` (switchable fitted/requested axis, stacked panels), `RunSeries` (time series with gap breaks and record links), `SelectionRanking` (thousands of points, histogram view), Blanks drift. Bar series: `bar: true` on a series draws each point as a bar (side by side on a point scale), and `histogramSeries` makes a histogram one series (a step outline, or `bars`). |
| Small multiples | `components/FacetGrid.tsx`, `lib/facets.ts` (`facetGrid`) | Blanks components, `KSweep`, every faceted `Figure`, and on the Weather tab `StationCheck`, `MonthlyClimate`, `RainWashout`. Replaces per-chart column guesses (`useFacetGrid` in §1). Optional chrome shared with composites: `title`/`note` per facet (the `.facet-title` row, kept in exports), `letters` (a, b, …; off unless asked for), ⤢ on by default (`expandable={false}` turns it off; draws a facet larger in a dialog), `maxCell`, `center`, and in a tile each row's share of the fitted height (`cell.height`). The facets are one panel group (below). |
| Composite figures (different charts in one frame) | `components/Panels.tsx` (`Panels`, `PanelCell`), `lib/facets.ts` (`panelRow`, `panelLetter`) | Weather `HumidityAgreement` (Crossplot + SweepPlot), `PwMoisture` (scatter + table), AERONET `AeronetTyping` (typing plane + class shares); `WeatherCorrelations` uses `PanelCell` for its title row and ⤢. Each panel declares `min`/`basis`/`max`/`grow`: side by side while every `min` fits, stacked below that, never a hand-made `flexWrap` with width arithmetic. Letters a, b, … on by default (two or more panels), readout at the right of the title row, ⤢ per panel (on by default), and in a tile or the one-chart view each panel gets its share of the height (also through `useFitHeight()`). Panels inside should be shared charts (`Crossplot`, `SweepPlot`, `SpectrumPlot`, …), not hand-drawn axes. Still hand-laid-out: the wind/pollution roses and polar plot (`roseParts` fits them to a tile), `CompositionTreemap`, and multi-svg page sections outside `charts/`. |
| Declared figures and meetings | `components/Figure.tsx` (kinds `crossplot`, `spectrum`, `sweep`, `intervals`, `histogram`, `heatmap`, `bars`, `table`, `custom` via `registerFigure`), `lib/table.ts`, `lib/markdown.tsx`, `pages/MeetingSpecPage.tsx`, `src/meetings/*.meeting.ts`, `gallery/data/meeting_bundle.py` | A meeting page is an exporter writing long tables and answer text plus a spec file (gallery/README.md "Declaring a figure or a whole meeting"). First one: Ann 9 Oct (`ann-20261009`). Vega-Lite was considered and not used: it would drop the gallery's look, estimator switching, the sample drawer and whole-panel export. |
| Dashboard | `pages/DashboardPage.tsx`, `components/dashboard/` (`DashHeader`, `registry`, `extraTiles`), `lib/dashboard.ts` (layout engine, node-tested), `lib/chartFit.tsx`, `charts/summary/` (`StatTile`, `SiteCards`, `SiteSummary`), `charts/ranking/SiteRank.tsx` | Added 17 Oct 2026. Scopes Site / Compare / Network with named sheets; 12 × 12 grid fitted to the window, phones one column; tiles compact (⋯ menu) with per-tile legend modes; every tile setting in the URL. |
| Controls | `components/Controls.tsx` (`Select`, `Segmented`, `Toggle`, `sentence`) | Split out of `ChartFrame.tsx` (31 Oct 2026); `ChartFrame` re-exports them, so imports from `@/components/ChartFrame` keep working. Labels follow the page's estimator. |
| Site and season colours, colour tokens | `lib/siteColours.ts` (`siteColor`, `seasonColor`, `siteCodeColor`, `namedColor`, `colouring`, `colourGroups`), `lib/theme.ts` (`MONTHS`, `MONTH_LETTERS`, `prettyUnit`, `PREPROCESSING_COLOR`, `LOT_COLOR`, `AETH_CHANNEL_COLOR`, `AETH_SPOT_COLOR`, `VARIABLE_COLOR`, `SPARTAN_COLOR`, …), `lib/palettes.ts` (`TABLEAU`, `RESEARCH_SERIES`) | A site's colour is `siteColor(meta, name)` (or `siteCodeColor` by code), with a muted fallback; never `meta.sites.find(...)?.color`. A categorical series takes `TABLEAU` or `RESEARCH_SERIES`, never a Tableau hex typed in a chart. A colour that means one thing everywhere (a preprocessing, a lot, an aethalometer channel, a weather variable) is a named map in `lib/theme`. |
| Statistics | `lib/stats.ts` only, `lib/table.ts` (`rowsOf`, `num`), `lib/seasons.ts` (`shortSeason`) | Every statistic a page computes: describe (`mean`, `median`, `quantile`, `describe`, `boxStats`, `monthlyStats`), correlate (`pearson`, `spearman`, `corrMatrix`, `correlationP`, `steiger`), fit (`regression`, `ols`, `deming`, `york`, `theilSen`, …) and the HIPS / meteorology screens. Inputs are null-tolerant (non-finite values are skipped, pairs kept only where both are finite) and a minimum count is an option (`{ minN }`). `sections/research/common/stats` is gone. |
| Label room | `lib/fitLabel.ts` (`labelGutter`, `fitsWidth`, `clipText`, `textWidth`) | A gutter or a "does it fit" test is measured from the text in the chart's font; never characters × px. |
| Vertical grid | `components/Axes` `XAxis` `gridHeight` (and the grid's top) | Replaces hand-drawn vertical grid lines. |
| Guardrail band | `components/RefLines.tsx` `RefBand` | The calibration slope box, one style (accent at 0.06, clamped to the plot): CutoffSweep, KSweep, SlopeInterceptTrap, SpectraDumbbell. |
| Tooltips on marks that are not points | `hooks/useTooltip` `tip.bind(body)` | Spread the handlers on a bar, cell, band or segment; no `<title>` beside it. `lib/tooltips.test.mjs` fails on a file with the gallery tooltip and a native `<title>`, except those in its `ALLOWED` list (a `<title>` that only restores a clipped label). |
| Dialog | `components/Dialog.tsx` (`Dialog`, `NESTED_POPOVERS`) | One modal for every caller: portalled, backdrop click and Escape close it, focus moved in and kept inside, given back on close. While a nested popover (a Combobox list, the export Colors popover) is open, Escape closes that first. Users: the export, ratio, grouping, guardrail and component-set dialogs, the site picker, the command palette, the glossary, SubsetBar, the Correlogram and the dashboard tools. `FilterSpectrumPlot` keeps a native `<dialog>`. |
| Escape unpins | `lib/highlight` `useEscapeUnpin()` | A page whose clicked filter stays pinned on every chart calls it once: Escape lets the pin go. Meeting and Satoshi pages. |
| Bar rows | `components/BarRows.tsx` | Labelled horizontal bars: a screen's scores, a ranking, shares of a whole, two populations side by side, stacked segments. Gutter measured from the labels, values printed through `lib/format`, signed rows in the r colours, the shared `XAxis`. `Figure` `bars` draws one from a table. Users: AttributionStudy, ClassShares, TorStudy, Similarity, Satoshi, the research `ui` helpers. |
| General scatter | `components/ScatterPlot.tsx` | Two different quantities (not a 1:1 comparison, which stays `Crossplot`): each axis linear, log or time, or a band axis of categories; points through `ScatterLayer`, Filters → excluded applied, ranges fixed, from the data or `robustDomain`, `refX`/`refY`/`identity`, an optional live fit with its summary, axis-title pickers. Users: Baselines, Ext pilot, Baseline diagnostics, Blanks, Meeting, Satoshi, Experimental. |
| Findings | `components/Findings.tsx` (`FindingsList`) | A study's numbered short answers and a note under them, one look on every research page (studies, Blanks, Satoshi). |
| Panel group | `components/PanelGroup.tsx`, provided by `Panels`, `FacetGrid` and `PanelGrid` | What the panels of one figure share. `share` gives the shared charts inside (`Crossplot`, `SweepPlot`, `SpectrumPlot`) one axis range under the gallery's Panel axes setting (`pax`: Shared, Per site, Own); a shape per axis is allowed. Crossplots draw one fit grouping box for the group. `legend` is drawn once under the panels and exported; `show`/`order`/`pick` choose and order panels by key; `useGroupControl` hoists a chart's control into one row above the panels. `PanelGrid` is the research and meeting `Grid`. ⤢ is on by default everywhere; letters are on in `Panels` and opt-in in `FacetGrid` and `PanelGrid`. Never compute a shared domain by hand. |
| Stacked panels | `components/StackedPanels.tsx`, `lib/stacked.ts` (`stackGeometry`, `lineRuns`, `nearestIndex`), `components/DailyLanes.tsx` | Panels one above another on one x axis (time, linear or point): a y per panel, one cursor and tooltip across all, season shading, markers through every panel, heights shared out in a tile, one `<svg>` for export. `DailyLanes` is the AERONET and weather timelines' daily record above the filters' measurements. Users: AeronetTimeline, WeatherTimeline, Aethalometer DailyPanels and AgreementOverTime, CutoffSweep, Baseline comparison. |
| Panel choice | `lib/panelPick.ts` (`usePanelPick`), `components/PanelPicker.tsx` | Which panels a chart shows and in what order, in the link as `<chartId>.panels`. Users: WeatherTimeline, HipsTransfer, RainWashout, AgreementOverTime. |
| Page presets | `lib/samplePresets.ts`, a page's `samples` in `sections/research/pages.ts` (`SectionPage.samples`), `lib/markedRows` `useSampleSplit`, `sections/research/common/ui` `useInSamples` | Each Research page is presented on its own Samples (sites, calendar, dates, the export's exclusions): a visit opens on them, and the reader adds to them in the panel. Those picks stay with that page for the session, the rest of the gallery keeps its own set, and Reset goes back to the page's preset. A link that names any sample key wins whole; `smp=1` marks a link left on every default. Hand exclusions (`drop`, `keep`) hold everywhere. A new page declares `samples`, and at that preset its charts draw what the notebook printed. A chart follows Samples through `rows`, or `useSampleSplit` / `useInSamples` for its own per-filter items. |
| View state | `lib/url.ts` (`useSearchParam` with `legacy` keys; `useTileSetting` is now an alias), `lib/chartOptions.tsx` (`useChartOptions` with `legacy` keys per option; shared options `testSetOption`, `cohortOption`, `bcOption`, `speciesFamilyOption`, `fitGroupingOption`), `lib/siteFocus.tsx` (`usePageSite` with `SitePicker`, one `psite` key; `useSiteMode` with `components/SiteModeHead`) | A control that moves to a new key lists the old one in `legacy`, so old links open the same view. Pages that show one site at a time share `psite`, so the site stays in view between them. Pages with a comparison and a one-site analysis (Weather, AERONET, Aethalometer overview and filters) use `useSiteMode` and draw their heading with `SiteModeHead`. |

Export colors use the captured figure's site and season identities and rendered estimators. New
site-colored marks or series groups carry `data-export-site` (site name or code)
and `data-export-color` (the original paint). `Legend` supplies matching swatch
metadata. Season marks and bands carry `data-export-season` (the full metadata name)
and `data-export-color`. The active calendar supplies the month membership. Consolidation
requires matching season names and months, sums legend counts, and applies one shared
color; it leaves filters and fitted populations intact. Encodings of other variables can use `data-export-series` to keep their
colors independent of site overrides. `RegressionOverlay` tags each estimator's
line, band and equation; use those shared components for fitted charts. Colors
apply to the export clone, and the browser remembers export preferences. Standard mode
uses the original chart colors without discarding the saved Custom palette. Each
color row offers named swatches, its chart standard, a custom picker and hex input.
All color controls open together from the export dialog’s Colors popover; Escape
closes the popover first and returns focus to its trigger. The
recommendation button spaces colors in Lab, using D3 categorical palettes and an
HCL extension for larger selections, with contrast thresholds against white.

Not built yet from §1: `IntervalBar` and `selectedStyle`. (`filterMarkHandlers`
became `ScatterLayer`; `GuardrailBands` is `RefBand`; `correlationMatrix` is
`lib/stats.corrMatrix`.)

Follow-ups (31 Oct 2026, after the unify merges):

- Move the per-panel layout loops in `Scatterplot`, `CrossSiteTransfer`,
  `RunCrossplot` and `AnalogLab` onto `FacetGrid` with `share`.
- Build `Crossplot` on `ScatterPlot`'s shared internals.
- Split `components/SubsetBar.tsx` and `lib/export.ts`.
- Move the remaining `tip.hide` sites that are filter points into `ScatterLayer`.
- Move `RainWashout`, `AgreementOverTime`, `WeatherTimeline` and `HipsTransfer`
  onto the layouts' `pick` prop (they call `usePanelPick` and lay out the
  panels themselves).
- A top x axis for `StackedPanels` (needs an orientation on `XAxis`).
- `XAxis`/`YAxis`: `tickValues` and log support (for `AeronetScaleHeightTime`),
  and a tick colour (for the `MonthlyClimate` dual axes).

## 5. Decisions (1 Oct 2026)

- **Baseline colours**: AIRSpec `#2C6E9E`, archived VIBES `#7A4FA3`, and one new
  distinct colour for full-range VIBES. `#2171b5` means only accent and OLS.
- **Precision**: concentrations and loadings 2 dp with unit; slope, intercept
  and R² 3 dp; equations as `equation()` writes them (`Deming y = 0.950x − 1.230`).
- **IQR minimum n**: 4, as MonthlyBand already does (default, not yet confirmed).
- **Research-page crossplots** keep showing every point ("Data" axes); the
  shared `Crossplot` takes the axes mode as a prop.
- **Export** saves the whole panel: every facet plus the legend in one SVG.
- **Dark theme (23 Oct 2026; CSS since 27 Oct)**: header button cycles light →
  dark → auto (follow the system); light stays the default. Marks are always
  drawn in the light colours (`INK`, `BASELINE_COLOR`, `SPECTRAL_REGION`, … in
  `lib/theme.ts` never change), and `lib/themeCss` maps each light colour that
  has a dark twin (a paint pair) in CSS: SVG `fill`/`stroke`/`stop-color`
  attributes of that hex, and inline HTML `color`/`background` of it, show the
  twin on `html[data-theme="dark"]` or in a `.panel[data-chart-theme="dark"]`.
  So a switch recolours the page in place, with no remount. Give a one-off
  colour its dark step with `themed(light, dark)` (it returns `light` and
  registers the pair). Only what CSS cannot map reads the theme and re-renders
  through `useTheme()` (lib/colorTheme): colour-scheme ramps (`useColourScheme`
  does it for you), `inkOn` labels on cells (`HeatCell` does it), canvas
  rasters (use `inkNow(key)`) and colours computed from the theme. Use
  `INK.panel` rather than `#fff` for halos, hollow markers and separators, and
  `inkOn` for cell labels (its dark ink is `CELL_INK`, which the CSS leaves
  alone). SVG `<text>` with no `fill` of its own and none inherited from a
  `[fill]` ancestor would be the browser's black; `lib/themeCss` `textRules()`
  gives it the dark text ink (1 Nov 2026), at zero specificity so any explicit
  fill still wins. Prefer `fill={INK.text}` / `INK.muted` in new code anyway,
  and never a grey hex without a `themed()` twin. A filled scatter dot that is
  not focused wears `dotRim()` from `lib/highlight` (the panel colour at 0.8 px,
  as `focusStyle` does), never a rim in its own colour or a dark hex, which shows
  as a bright ring or disappears on the dark panel (27 Oct 2026). Hollow dots,
  rims that encode something, and sub-2.5 px dots in dense clouds keep their own
  stroke. Its fill opacity is a level, not a number: `dotFill(level)` or
  `focusStyle(..., { r, level })` with `sparse` 0.85, `standard` 0.7, `dense`
  0.45 or `dimmed` 0.25 (`DOT_OPACITY` in `lib/theme`). The dark page lifts the
  first three (0.92 / 0.82 / 0.6) in CSS through the mark's `data-dot` tag, so
  hues stay apart; leave the tag off a dot scaled by a dim. Site and season
  colours stay the same in both themes. Exports are always light: SVG and PNG
  carry no stylesheet, the PDF's off-screen copy sits in `[data-print]`, which
  the CSS skips, and `lib/export` still maps colours read from the page (HTML
  legend text, dark ramp cells) back to their light step.
