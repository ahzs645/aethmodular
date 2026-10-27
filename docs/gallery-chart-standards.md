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

---

## 1. Standards, one per feature

Each row is the rule we standardise on, the primitive that carries it, and how
many places currently deviate. **New** marks a primitive that does not exist yet.

| Feature | Standard | Primitive | Deviations |
|---|---|---|---|
| Panel, title, explain, export | Every chart sits in `ChartFrame` | `components/ChartFrame` | 0, but export takes only the first `<svg>` (see §3) |
| Controls | `Select` / `Segmented` / `Toggle`, never raw `<select>` or checkbox | `components/ChartFrame` | 10 raw controls (Baselines ×7, Meeting ×3) |
| Sizing | `useDimensions`; square crossplots ≤ 560 px, min 240 | `hooks/useGalleryData` + **new** `useFacetGrid` | Hexbin 520; Ext pilot fixed viewBox ×3; facet maths copied ×4 |
| Margins | `MARGIN` unless the chart needs a label gutter | `lib/theme` + **new** `labelGutter(strings)` | 9 literal margins; 7 char-width gutter guesses (6.1–8 px/char) |
| Axes, ticks, titles | `XAxis` / `YAxis` (11 px ticks, 12/600 titles); band scales for category rows | `components/Axes`, add `gridHeight` to `XAxis` | ~12 hand-drawn axes or row labels; 4 hand-drawn vertical grids |
| Domain | `pairDomain` for crossplots; **new** `robustDomain(vals, lo, hi)` returning clamp + pinned count for clipped charts | `lib/axes` | 6 calibration variants (0.5–99.5 %, 2–98 %, p99, hard caps); 6 gallery quantile clips |
| 1:1, zero, one, marker lines | One style per kind (identity `5 4` w1.5 `INK.identity`; zero/one solid `INK.axis` .7; marker dashed `3 3` + mono 10 label) | **new** `RefLine`, `IdentityLine`, `Marker` | ~25 hand-drawn lines with 5 dash patterns |
| Fit lines and equations | `RegressionLines` for every fit, saved or live; `equation()` for text (3 dp, "y =") | `components/RegressionOverlay`, `lib/regressionPrefs` | CrossSiteTransfer clips by hand; 3 equation formats (`fmtFit` 2 dp, `equation` 3 dp, inline strings) |
| Fit summary placement | Above the axes at every width: n / R², then one equation per line; group summaries stack on phones and use up to two columns on wider panels. Wrap long text at a fixed readable size and retain it in exports. | `components/RegressionOverlay.useRegressionSummary` | Applied to scatter, hexbin, shared research crossplots, station comparisons, and filter drawer crossplots. Specialised saved-result legends retain their own layout. |
| Intervals | Whisker bars share one style | **new** `IntervalBar` (beside `RegressionBand`) | HipsYork, CrossSiteTransfer, Meeting ×2, BC LoadingPlot: 3 whisker styles |
| Guardrails | Slope box and held-out floor drawn one way | **new** `GuardrailBands` in `charts/calibration/common` | 4 slope boxes, 3 floor styles |
| Tooltip | `useTooltip` only; no native `<title>` alongside it | `hooks/useTooltip` | 5 charts show both tooltips; ~10 show only native `<title>` |
| Tooltip content | `id`, `site · date`, `name: value unit`, trailing "click for details" | **new** `filterMarkHandlers(row, lines)` next to `useHighlight` | 7 copies of hover + tip + open; 4 wordings (`x: v`, `x = v`, `≈`, `median =`) |
| Legend | `Legend` / `useLegend` as the HTML row inside `.chart-wrap`; `ColorLegend` for every ramp | `components/Legend`, `ColorLegend`; extend `useLegend` with background keys (seasons) and an in-SVG mode for export | 3 legends bypass `useLegend`; ~10 in-SVG one-offs; Meeting `Key`; FilterSpectrum div legend; 3 legends outside `.chart-wrap` |
| Highlight, pin, dim | `focusStyle` for filter marks; **new** `selectedStyle` for a selected configuration | `lib/highlight` | Meeting Scatter, ParallelCoordinates, DistributionPanel outliers; 6 selected-config styles |
| Brush | One accent brush style | **new** `useD3Brush` | Scatterplot uses d3 grey; ParallelCoordinates .18; DateBrush .14 |
| Season shading | One component for bands / strips / arcs | **new** `SeasonBands` | 5 implementations |
| Colours | Sites and seasons from `meta`; INK tokens; **new** `METHOD_COLOR` for baselines keyed like `PREPROCESSING`; **new** region-shade token | `lib/theme`, `lib/labels` | AIRSpec/VIBES defined 7×, 3 different colours for full-range VIBES; INK values typed as hex in ≥15 places |
| Heatmap cells | Luminance-based text colour; always a `ColorLegend` | **new** `HeatCell` | Hand-tuned thresholds ×2; SlopeInterceptTrap and SeasonMatrix have no colour legend |
| Numbers | `fmt` (null → "—"); **new** `fmtSigned`, `fmtPct`; one unit prettifier | `lib/stats`, `lib/theme.withUnit` | 6 local formatters (`f`, `f2`, `f3`, local `fmt`); ~25 `?.toFixed(3) ?? '—'`; 3 unit prettifiers |
| Statistics | Shared helpers for repeated maths | **new** `monthlyStats`, `correlationMatrix` in `lib/stats` | IQR at n≥4 (MonthlyBand) vs n≥3 (SeasonalClock); correlation matrix ×3 |
| Data loading | One fetch + schema check + loading/error panel | **new** `useExport(url, validate)` | 6 copies; 4 pages show a bare "Loading…" outside any panel |
| Empty state | `Empty`, and the chart is not drawn behind it | `components/ChartFrame` | AvailabilityHeatmap none; Treemap and BubbleMap draw an empty chart under it |
| Transitions | `.animated` class | `styles.css` | 5 charts snap; 3 use inline transitions |
| View state | `useSearchParam` for anything a link should reopen | `lib/url` | ~8 `useState` controls on research pages |
| Height in a tile | The chart's height (or row height, or square side) from `useFitHeight() ?? its own`; legends through `Legend` so a tile can compact, move or hide them | `lib/chartFit`, `components/Legend` | Wind roses (fixed-size roses) and multi-panel layouts (per-site grids) keep their size and scroll inside the tile |
| Dashboard tile | One `ChartFrame` per chart; registered in the Add chart menu; settings in `useChartOptions` / `useTileSetting` so a shared dashboard reopens them | `components/dashboard/registry`, `lib/url` | The Weather and Calibration charts join through adapters (`components/dashboard/extraTiles`) |
| Small constants | `MONTHS`, heat ramps defined once | `lib/theme` | `MONTHS` ×3, `HEAT` ×2 |
| Floating layers (popovers, notes, menus) | Portalled to `<body>`, `position: fixed`, placed from the trigger and kept inside the window; never `position: absolute` inside a panel, where a scrolling box, a tile or a phone's edge cuts it off | `hooks/useFloatingMenu` (menus, pickers), `ChartFrame` `InfoTip` / `useFloatingTip` (notes), `hooks/useTooltip` (hover readouts) | Header-only popovers (Views, nav menus, Dashboard header panels) still sit in the header; `src/lib/floating.test.mjs` fails on any new in-place popover |
| Stepping (previous / next) | One chevron button, square and as tall as the `Combobox` beside it; no text arrows, no taller touch buttons beside a picker | `components/StepButton` | 0 (single-chart view, Connected scatter) |

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
  This is why several pages hand-draw legends inside the SVG.
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

## 4b. Status (14 Oct 2026)

Done, and each defined once in `gallery/app/src`:

| Feature | Lives in | Notes |
|---|---|---|
| Baseline colours (`BASELINE_COLOR`, `baselineColor(method, rank)`), IMPROVE grey, Addis-all yellow, bank colours, spectral regions, pinned/hovered trace style | `lib/theme.ts` | The Baselines exports call every VIBES run "VIBES"; `lib/baselineRank.tsx` gives each view its run's rank so rank-70 VIBES is orange on every tab. `#2171b5` is now only accent and OLS. |
| Baseline names, short names, EC axis titles | `lib/labels.ts` (`baselineName`, `PREPROCESSING_SHORT`, `ecAxis`) | |
| Number formats | `lib/format.ts` (`fmt`, `f2`, `f3`, `signed`, `pct`, `sci`, `fitLine`, `coef`, `coefSigned`) | `lib/stats.fmt` re-exports it. The §5 precision is applied (11 Oct): slopes, intercepts and R² at 3 dp through `coef`/`coefSigned`, `FitCell`/`BandCell` and `fmtFit`; concentrations stay at 2 dp. |
| Seasons | `lib/seasons.ts` | Two Addis calendars on purpose (docs/site-seasonality.md): pages showing frozen calibrations use `dry_feb` and say so (`ANALYSIS_CALENDAR_NOTE`); names are normalised to one dash. |
| Data loading | `lib/data.ts` (`loadJson`, `loadBinary`, `useExport`), `components/ExportState` | One cache: every file downloads once per session. Spectra: `lib/similarity.loadTraces` checks the grid against the filter table for every page and the drawer. |
| Pickers | `components/Pickers.tsx` (`BaselinePicker`, `SeasonPicker`, `BankPicker`) | Page option lists come from data or context, not module `let`. Raw `<select>`/checkbox controls replaced. |
| URL keys | `lib/url.ts` | Development warning when two controls use one key with different options; the `sregion` collision is fixed. |
| Crossplot | `components/Crossplot.tsx` | Meeting, Satoshi (`Cross`, `Envelope` with a density `background`), `RunCrossplot` (residual panel kept beside it on the same x), Ext `AllFilterScatter` optical panel (its own `grouping`). The Ext band-height panel is not a 1:1 comparison and keeps its own axes. |
| Static key | `components/Key.tsx` | Meeting and Satoshi; in-SVG legends kept where export needs them. |
| Spectrum shading | `components/SpectralRegions.tsx` | Meeting, Satoshi VIP, Blanks. The Baselines amber band marks the earlier selection masks (a different meaning) and stays. |
| Dot-interval rows | `components/IntervalStrip.tsx` | Meeting `RepeatStrip`, `Combined`; Satoshi `Swap`, `BlankEffect`; Baselines `LoadingPlot`; `HipsYork`; Similarity `GroupRanking` (stems, selectable rows, labelled marker). `CrossSiteTransfer` ordering is a jittered dot plot with per-dot picking, a different chart, and stays. |
| Heat-map cells | `lib/theme.inkOn` (contrast-based label ink), `components/HeatCell.tsx` | Label ink in Correlogram, CrossSiteHeatmap, OverlapMatrix and the Similarity season matrix (which now has its `ColorLegend`). |
| Table cells | `components/TableCells.tsx` (`FitCell`, `BandCell`) | Meeting and Satoshi fit and median-interval cells. |
| Reference lines | `components/RefLines.tsx` (`IdentityLine`, `RefLine`, `Marker`) | 1:1 lines, slope-1 / intercept-0 / zero lines and labelled markers on every calibration, correlation, Meeting, Satoshi, Baselines and Blanks chart (39 hand-drawn lines). Crosshairs, the time cursor, stems and fit lines are not reference lines and keep their own styles. |
| Clipped domains | `lib/axes.ts` (`robustDomain(values, lo, hi, { include, limit, fallback })`, returns the domain and the counts above and below) | SpectraDumbbell, SlopeInterceptTrap, CutoffSweep, SelectionRanking, SpecCurvePlots, Scatterplot, Correlogram, Ridgeline, Histogram, ParallelCoordinates, MonthlyHeatmap, Meeting slope-intercept map, Satoshi ladder and envelope, Blanks raw spectra. Each keeps its own quantiles; they now share one implementation and one pinned count. |
| Season shading | `components/SeasonBands.tsx` (`SEASON_SHADE`, `seasonShade`, `timeSeasonBands`, `monthSeasonBands`, `<SeasonBands>`) | Timeseries, MonthlyBand, SourceStack (strip), SeasonalClock arcs, MonthlyHeatmap strips. One set of opacities, including the legend-hover deepen/fade. |
| Brush | `lib/brush.ts` (`BRUSH_STYLE`, `styleBrush`) | Scatterplot, ParallelCoordinates, DateBrush: accent fill .14, handles .6, no transition. A style helper rather than a hook, because each chart owns its d3 brush lifecycle. |
| Export | `lib/export.ts` (`composeFigure`, `layoutFigure`, `renderFigure`), `components/ExportDialog.tsx`, `lib/exportColors.ts` | PNG, JPEG, SVG and PDF use the same composed figure and color overrides, including every facet, legend and readout. Site colors can be individual or shared; season colors can be separate by site or consolidated by matching names and months; estimator overrides also color equations and confidence bands. Mark other HTML text with `data-export="text"` to include it. |
| Spectrum chart | `components/SpectrumPlot.tsx` (helpers `lib/spectra.ts`) | Meeting `SpectraPanel` and `BaselineCompare`; Blanks raw spectra, variability, PCA inset, per-blank error inset, components, after correction; Satoshi `Vip` and fingerprint panel; calibration `SpectraOverlay`; full-range VIBES aggregate spectra. Per-series wavenumber grids, one hover mechanism (`nearest` names a trace and pins it, `readout` lists every line), pin linked to the page with `highlight={useHighlight()}`, `markers`, `background`/`children` render props. Left on their own on purpose: `FilterSpectrum` (zoom and pan), `SimilaritySpectra` (shading from a channel mask), Baselines `SpectraPlot` (difference and hidden-mask modes). |
| Sweep chart | `components/SweepPlot.tsx` | Calibration `CVCurve`, `KSweep` (three facets); Blanks rank selection and capture; Satoshi ladder (point scale with a divider) and blank-bank rank CV; Meeting MAC envelope (log x); full-range VIBES injection recovery. Bands or whiskers, the agreement value, labelled markers, the chosen setting ringed, clipped points hollow at the edge. Left on their own: `CutoffSweep` (switchable fitted/requested axis, stacked panels), `RunSeries` (time series with gap breaks and record links), `SelectionRanking` (thousands of points, histogram view), Blanks drift. |
| Small multiples | `components/FacetGrid.tsx`, `lib/facets.ts` (`facetGrid`) | Blanks components, `KSweep`, and every faceted `Figure`. Replaces per-chart column guesses (`useFacetGrid` in §1). |
| Declared figures and meetings | `components/Figure.tsx` (kinds `crossplot`, `spectrum`, `sweep`, `intervals`, `histogram`, `heatmap`, `table`, `custom` via `registerFigure`), `lib/table.ts`, `lib/markdown.tsx`, `pages/MeetingSpecPage.tsx`, `src/meetings/*.meeting.ts`, `gallery/data/meeting_bundle.py` | A meeting page is an exporter writing long tables and answer text plus a spec file (gallery/README.md "Declaring a figure or a whole meeting"). First one: Ann 9 Oct (`ann-20261009`). Vega-Lite was considered and not used: it would drop the gallery's look, estimator switching, the sample drawer and whole-panel export. |

| Dashboard | `pages/DashboardPage.tsx`, `components/dashboard/` (`DashHeader`, `registry`, `extraTiles`), `lib/dashboard.ts` (layout engine, node-tested), `lib/chartFit.tsx`, `charts/summary/` (`StatTile`, `SiteCards`, `SiteSummary`), `charts/ranking/SiteRank.tsx` | Added 17 Oct 2026. Scopes Site / Compare / Network with named sheets; 12 × 12 grid fitted to the window, phones one column; tiles compact (⋯ menu) with per-tile legend modes; every tile setting in the URL. |

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

Not built yet from §1: `labelGutter`, `XAxis gridHeight`,
`IntervalBar`, `GuardrailBands`, `filterMarkHandlers` (and the tooltip
de-duplication it carries), `selectedStyle`, `monthlyStats` /
`correlationMatrix`.

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
  alone). A filled scatter dot that is
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
