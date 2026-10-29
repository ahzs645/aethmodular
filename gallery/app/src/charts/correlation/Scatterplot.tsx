import { fitUncertainty } from '@/lib/fitUncertainty'
import { styleBrush } from '@/lib/brush'
import { seasonsForSite } from '@/siteSeasons'
import { IdentityLine, RefLine } from '@/components/RefLines'
import { RegressionLines, RegressionGroupLines, useRegressionSummary } from '@/components/RegressionOverlay'
import { useRegressions, fitsFor } from '@/lib/regressionPrefs'
import { fitGroupingOption, useChartOptions, type OptionContext } from '@/lib/chartOptions'
import { groupedRegressions, type FitGrouping, type RegressionGroup } from '@/lib/siteRegression'
import { memo, useCallback, useEffect, useId, useMemo, useRef } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { ColorLegend, SizeLegend } from '@/components/ColorLegend'
import { FieldSelect } from '@/components/FieldSelect'
import { Legend, toggleIn, useLegend, type LegendItem } from '@/components/Legend'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { PanelAxesControl } from '@/lib/panelAxes'
import { useTooltip } from '@/hooks/useTooltip'
import { FocusLayer, focusStyle, useHighlight } from '@/lib/highlight'
import { useFilterEdits } from '@/lib/filterEdits'
import { inDomain, pairDomain, panelAxesKey, type AxesOpts, type PairDomain, robustDomain } from '@/lib/axes'
import { regression, fmt, type RegressionStats } from '@/lib/stats'
import { INK, MARGIN, FONT, withUnit } from '@/lib/theme'
import { useColourScheme } from '@/lib/colourSchemes'
import type { FilterRow, MetaFile } from '@/lib/types'
import { NO_PMF, PMF_SOURCE } from '@/lib/pmfGroups'
import { ChartSite, useChartLabel, useFitHeight } from '@/lib/chartFit'
import { useSearchParam, useTileSetting } from '@/lib/url'
import { GroupingPicker, useGroupings } from '@/components/GroupingDialog'
import { NO_BAND_COLOR, applyGrouping, noBandLabel } from '@/lib/groupings'
import { ChartAside } from '@/lib/singleView'
import { ScrubField } from '@/components/ScrubField'
import { UndoIcon } from '@/components/Icons'

type Point = { row: FilterRow; x: number; y: number; c: number | null; s: number | null }
type Scale = d3.ScaleLinear<number, number> | d3.ScaleLogarithmic<number, number>

/** Per season: one site's seasons side by side, or each site's seasons grouped under it */
const LAYOUTS = ['Combined', 'Per site', 'Per season', 'Per group'] as const
const COLOR_BY = ['Site', 'Season', 'Value', 'Group'] as const
/** layouts drawn as one row of panels per site (its seasons, or the bands of a grouping) */
const ROWED = (layout: unknown) => layout === 'Per season' || layout === 'Per group'
/** Value's categorical choice where a site in view has PMF: each point by its day's dominant source */
const PMF_COLOUR = 'PMF source (dominant)'
const NO_SIZE = 'none'
/** several panels drawn at a fixed plot side (px), dragged or typed, instead of fitted to the chart or the window */
const PANEL_MIN = 120
const PANEL_MAX = 900

/**
 * Sequential ramp over the 2–98 % bulk of a third variable, so one outlier does not wash the rest to the first colour.
 * The palette's two palest stops are left out (whichever end they are at): a near-white point vanishes on the plot.
 */
function valueScale(values: number[], stops: string[]) {
  const [lo, hi] = robustDomain(values, 0.02, 0.98).domain
  const paleFirst = d3.lab(stops[0]).l > d3.lab(stops[stops.length - 1]).l
  const ramp = paleFirst ? stops.slice(2) : stops.slice(0, -2)
  return d3.scaleSequential(d3.interpolateRgbBasis(ramp)).domain([lo, hi === lo ? lo + 1 : hi]).clamp(true)
}

/**
 * The crossplot family — 281 of the 827 figures in this repo, by far the
 * largest single group (see gallery/census/CENSUS.md), plus the 123 that
 * colour the same crossplot by a third measured quantity ("Iron colour-coded",
 * "Fe/BC ratio"). That third variable is the `Value` colour mode here, and a
 * fourth can go into the mark size (react-graph-gallery's bubble plot).
 *
 * Interactive counterpart of `plotting/crossplots.py`, including its
 * PlotConfig.layout: "Combined" overlays every site on one axes, "Per site"
 * is the 2×2 grid, with the axis domain shared across panels so the eye can
 * compare slopes. The sidebar selects OLS, Deming and York; Deming is the
 * default because both axes carry measurement uncertainty.
 *
 * Hover uses closest-point detection (d3.Delaunay) over the whole panel
 * rather than per-circle mouse events, so a dense cloud still answers the
 * pointer and a 2 px mark is as easy to reach as a 12 px one. With `brush to
 * select` on, dragging a rectangle puts the enclosed filters into the shared
 * selection (lib/highlight) and every chart on the tab dims the rest — the
 * "brush a cloud of points, see them on the timeseries" step the README
 * listed as the next one after hover/pin.
 *
 * The 1:1 toggle and the axes mode live in the subset bar so the hexbin and
 * the connected scatter below draw the same frame.
 */
export function Scatterplot({ rows, meta, xField, yField, axes, id = 'scatter', title = 'Scatterplot — the crossplot family' }: {
  rows: FilterRow[]; meta: MetaFile; xField: string; yField: string; axes: AxesOpts
  /** another tab reusing the crossplot (Weather) names it for what it shows */
  id?: string; title?: string
}) {
  const chartLabel = useChartLabel()
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  // the chart's own state (pins, brush, handlers) without re-rendering on hovers; the hover is read below, for this
  // chart's filters only
  const hl = useHighlight(NO_KEYS)
  const { methods, grouping: chosenGrouping, setGrouping } = useRegressions()
  const { hidden, setHidden, dim, hover, setHover } = useLegend()
  // the reader's colour field, or Iron, or the first field, among the fields this view has now: the
  // chart can mount while a network's fields are still loading (empty), or keep a field the next network lacks
  const [colorParam, setColorParam] = useTileSetting<string>('scatter.colorField', '')
  const colorChoice = colorParam || null
  const setColorField = (f: string | null) => setColorParam(f ?? '')
  const hasPmf = !!meta.pmf_groups?.length && rows.some((r) => typeof r[PMF_SOURCE] === 'string')
  const colorField = [colorChoice, 'Iron', meta.fields[0]].find((f): f is string => !!f && (meta.fields.includes(f) || (f === PMF_COLOUR && hasPmf))) ?? ''
  const [sizeField, setSizeField] = useTileSetting<string>('scatter.size', NO_SIZE)
  const nSites = useMemo(() => new Set(rows.map((r) => r.site)).size, [rows])
  // the options and when each applies (lib/chartOptions). Fits are per panel, so the grouping
  // offered follows the layout: a fit per site inside a site's panel is the pooled fit, and a
  // per-season panel holds one site in one season, so every grouping is its one fit
  // the reader's grouping (lib/groupings) for the Per group layout and the Group colour: its id, kept in the URL
  const groupings = useGroupings()
  const [groupId, setGroupId] = useSearchParam<string>(`${id}.group`, '')
  // the panel size in px, '' for Auto
  const [panelParam, setPanelParam] = useSearchParam<string>(`${id}.panel`, '')
  const groupingDef = groupings.find(groupId)
  const noBand = groupingDef ? noBandLabel(groupingDef) : 'No value'
  const o = useChartOptions('scatter', [
    // one site in view: no per-site grid, and per-season panels colour by season (one site colour says nothing)
    { key: 'layout', label: 'Layout', choices: LAYOUTS.map((v) => ({ value: v, when: v === 'Per site' ? (c: OptionContext) => (c.sites as number) > 1 : v === 'Per group' ? (c: OptionContext) => !!c.hasGroup : undefined })),
      default: 'Combined', title: 'PlotConfig.layout: combined overlays sites; per site is the grid', onChange: () => setHidden(new Set()) },
    // per-season panels: each site's seasons in one row that scrolls sideways, or wrapped onto as many lines as fit
    { key: 'seasonRows', label: 'Panels', choices: [{ value: 'line', label: 'One row' }, { value: 'wrap', label: 'Wrap' }], default: 'line',
      when: (c) => ROWED(c.layout), title: 'One row: each site\'s seasons side by side, scrolling sideways when they do not fit · Wrap: as many panels per line as fit the chart, the rest on the next line' },
    // several panels bigger than they fit: a fixed side per plot, the chart scrolls sideways and the page down
    // the field shows the fitted size while on Auto, so a drag starts from what is drawn; ↺ goes back to Auto
    { key: 'panelSize', when: (c) => c.layout !== 'Combined', node: () => <span className="control panel-size"
      title="Every plot's side in px: drag sideways or click to type. Panels that do not fit scroll sideways inside the chart and down the page. ↺ goes back to Auto, the panels fitted to the chart (in the one-chart view, the window)">
      Panel size
      <span className="panel-size-row">
        <button type="button" className="btn quiet icon-btn panel-size-reset" aria-label="Panel size: back to Auto" title={sized ? 'Back to Auto: the panels fit the chart' : 'Auto: the panels fit the chart'}
          disabled={!sized} onClick={() => setPanelParam('')}><UndoIcon /></button>
        <ScrubField label="Panel size" value={sized || Math.round(innerW)} min={PANEL_MIN} max={PANEL_MAX} step={10} unit={sized ? 'px' : 'auto'} width={110}
          onChange={(v) => setPanelParam(String(Math.round(v)))} />
      </span>
    </span> },
    { key: 'colorBy', label: 'Colour', choices: COLOR_BY.map((v) => ({ value: v, when: v === 'Site' ? (c: OptionContext) => !(c.layout === 'Per season' && (c.sites as number) <= 1) : v === 'Group' ? (c: OptionContext) => !!c.hasGroup : undefined })), default: 'Site', title: 'Value: a third measurement on a sequential ramp (2–98 % of its range), or, where a site in view has PMF, its day\'s dominant source', onChange: () => setHidden(new Set()) },
    // the grouping for Per group panels and the Group colour: always offered; those two choices appear once one is picked
    // the first pick (from none) colours by it at once; Per group stays a choice
    { key: 'groupBy', node: () => <GroupingPicker value={groupId} rows={drawable} onChange={(id) => { if (!groupingDef && id) o.set('colorBy', 'Group'); setGroupId(id) }} /> },
    // where a site in view has PMF (lib/pmfGroups; Addis so far), its dominant source heads the list, beside the PMF shares
    { key: 'colorField', when: (c) => c.colorBy === 'Value', node: () => <FieldSelect label="by" value={colorField} meta={meta} onChange={(f) => { setColorField(f); setHidden(new Set()) }} extraOptions={hasPmf ? [PMF_COLOUR] : []} /> },
    { key: 'size', node: () => <FieldSelect label="Size" value={sizeField} meta={meta} onChange={setSizeField} extraOptions={[NO_SIZE]} title="Bubble plot: mark area proportional to a fourth measurement" /> },
    // several panels: the gallery's Panel axes (lib/panelAxes), here as well as in the sidebar
    { key: 'panelAxes', when: (c) => c.layout !== 'Combined', node: (c) => <PanelAxesControl dims={ROWED(c.layout) ? ((c.sites as number) > 1 ? ['site', 'season'] : ['season']) : ['site']} /> },
    { key: 'fitLines', label: 'Fit lines', toggle: true, default: true },
    fitGroupingOption({ bind: [chosenGrouping, setGrouping as (v: never) => void],
      when: (c) => !!c.fitLines && (c.estimators as number) > 0 && !ROWED(c.layout),
      hiddenNote: (c) => (c.fitLines && (c.estimators as number) > 0 && ROWED(c.layout) ? 'one fit per panel' : null),
      offerSite: (c) => c.layout === 'Combined',
      pooledLabel: (c) => (c.layout === 'Per site' ? 'One fit per site' : 'One pooled fit') }),
    { key: 'brushing', label: 'Brush to select', toggle: true, default: false, title: 'Drag a rectangle: the enclosed filters are highlighted on every chart of this tab', onChange: ((v: boolean) => { if (!v) hl.setSelected(null) }) as (v: never) => void },
    { key: 'selected', when: () => !!hl.selected, node: () => <span className="control"><strong>{hl.selected?.size}</strong> selected<button type="button" className="btn quiet" onClick={() => hl.setSelected(null)}>Clear</button></span> },
    { key: 'count', node: () => <span className="control">{points.length} points</span> },
  ], { estimators: methods.length, sites: nSites, hasGroup: !!groupingDef })
  const layout = o.values.layout as (typeof LAYOUTS)[number]
  const colorBy = o.values.colorBy as (typeof COLOR_BY)[number]
  const showFit = o.values.fitLines as boolean
  const brushing = o.values.brushing as boolean
  const grouping = o.values.grouping as FitGrouping
  const { log: logAxes } = axes

  const seasonColor = useMemo(() => new Map(meta.seasons.map((s) => [s.name, s.color])), [meta.seasons])
  const pmfColor = useMemo(() => new Map((meta.pmf_groups ?? []).map((g) => [g.name, g.color])), [meta.pmf_groups])
  const pmfOf = (r: FilterRow) => (typeof r[PMF_SOURCE] === 'string' ? (r[PMF_SOURCE] as string) : NO_PMF)
  // colour by PMF source is categorical: its legend groups (and hides) by source
  const pmfColouring = colorBy === 'Value' && colorField === PMF_COLOUR
  const groupOf = (r: FilterRow) => (colorBy === 'Season' ? r.season : colorBy === 'Group' ? grp?.labelOf(r) ?? noBand : pmfColouring ? pmfOf(r) : r.site)
  const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)
  // the grouping on the filters this chart can draw (both axes measured): its quantiles do not move when a band is
  // hidden in the legend
  const drawable = useMemo(() => rows.filter((r) => num(r[xField]) !== null && num(r[yField]) !== null), [rows, xField, yField]) // eslint-disable-line react-hooks/exhaustive-deps
  const grp = useMemo(() => (groupingDef ? applyGrouping(groupingDef, drawable) : null), [groupingDef, drawable])

  const points = useMemo<Point[]>(
    () =>
      rows
        .filter((r) => !hidden.has(groupOf(r)))
        .map((r) => ({ row: r, x: r[xField] as number, y: r[yField] as number, c: colorBy === 'Value' ? num(r[colorField]) : null, s: sizeField !== NO_SIZE ? num(r[sizeField]) : null }))
        .filter(
          (p) =>
            typeof p.x === 'number' && typeof p.y === 'number' &&
            Number.isFinite(p.x) && Number.isFinite(p.y) &&
            (!logAxes || (p.x > 0 && p.y > 0))
        ),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [rows, xField, yField, logAxes, hidden, colorBy, colorField, sizeField, grp]
  )

  // third variable → colour; fourth → size (sqrt, so area is proportional)
  // the ramp's palette (lib/colourSchemes): chosen by clicking the ramp in the legend
  const colours = useColourScheme('scatter', 'sequential')
  const cScale = useMemo(() => (colorBy === 'Value' && !pmfColouring ? valueScale(points.map((p) => p.c).filter((v): v is number => v !== null), colours.stops) : null),
    [points, colorBy, pmfColouring, colours.stops.join()]) // eslint-disable-line react-hooks/exhaustive-deps
  const sScale = useMemo(() => {
    if (sizeField === NO_SIZE) return null
    const vals = points.map((p) => p.s).filter((v): v is number => v !== null && v > 0)
    return d3.scaleSqrt().domain([0, d3.quantile(vals.slice().sort(d3.ascending), 0.98) ?? 1]).range([1.5, 9]).clamp(true)
  }, [points, sizeField])
  const nNoColor = colorBy === 'Value' ? points.filter((p) => p.c === null).length : 0
  const nNoSize = sScale ? points.filter((p) => p.s === null).length : 0

  // stable between renders (they change only with the colouring, the size field or the legend), so a panel whose
  // own inputs did not change skips re-rendering (ScatterPanel is memoised)
  const colorOf = useCallback((p: Point) =>
    pmfColouring ? pmfColor.get(pmfOf(p.row)) ?? INK.neutral
      : colorBy === 'Value' ? (p.c === null || !cScale ? INK.neutral : cScale(p.c)) : colorBy === 'Season' ? seasonColor.get(p.row.season) ?? INK.muted
      : colorBy === 'Group' ? (grp ? grp.colorOf(grp.labelOf(p.row) ?? noBand) : INK.neutral)
      : (p.row.color as string), [pmfColouring, pmfColor, colorBy, cScale, seasonColor, grp]) // eslint-disable-line react-hooks/exhaustive-deps
  const radiusOf = useCallback((p: Point) => (sScale ? (p.s === null ? 1.5 : sScale(p.s)) : 4), [sScale])
  const dimOf = useCallback((p: Point) => dim(groupOf(p.row)), [hidden, hover, colorBy, pmfColouring, grp]) // eslint-disable-line react-hooks/exhaustive-deps

  const panels = useMemo(() => {
    const raw: { key: string; title: string | null; color: string; points: Point[]; group?: string; site?: string }[] = layout === 'Combined'
      ? [{ key: 'all', title: null as string | null, color: INK.text, points }]
      : layout === 'Per group' && grp
        // each site's row: one panel per band of the grouping
        ? meta.sites.flatMap((s) => grp.labels.map((l, i) => ({
            key: `${s.code}|g${i}`, group: s.name, site: s.name, color: grp.colors[i], title: l,
            points: points.filter((p) => p.row.site === s.name && grp.labelOf(p.row) === l),
          }))).filter((p) => p.points.length > 0)
      : layout === 'Per season'
        // each site's own seasons under the active calendar (local calendars carry the site in the name)
        ? meta.sites.flatMap((s) => seasonsForSite(meta, s.name).map((se) => ({
            key: `${s.code}|${se.name}`, group: s.name, color: se.color, site: s.name,
            title: se.site && se.name.startsWith(`${se.site} · `) ? se.name.slice(se.site.length + 3) : se.name,
            points: points.filter((p) => p.row.site === s.name && p.row.season === se.name),
          }))).filter((p) => p.points.length > 0)
        : meta.sites
          .map((s) => ({ key: s.code, title: s.name, color: s.color, site: s.name, points: points.filter((p) => p.row.site === s.name) }))
          .filter((p) => p.points.length > 0)
    // Deming lambda: when both axes are the same measured quantity we default to
    // 1.0 (orthogonal), exactly as calculate_regression_stats does.
    return raw.map((p) => ({
      ...p,
      ids: new Set(p.points.map((q) => q.row.id)),
      stats: regression(p.points.map((q) => q.x), p.points.map((q) => q.y), fitUncertainty(p.points.map(q => q.row), xField, yField, meta)),
      fitGroups: groupedRegressions(p.points, grouping, { x: q => q.x, y: q => q.y, site: q => q.row.site, season: q => q.row.season, seasonColor: q => meta.seasons.find(se => se.name === q.row.season)?.color ?? INK.muted,
        color: q => meta.sites.find(s => s.name === q.row.site)?.color ?? INK.muted,
        options: members => fitUncertainty(members.map(q => q.row), xField, yField, meta) }),
    }))
  }, [layout, points, meta, xField, yField, grouping, grp])

  // Shared domain across every panel by default — a per-site grid with different axes
  // per panel silently hides that one site's slope is twice another's. "Own" panel axes
  // (sidebar) fit each site's panel to its own points, still square when 1:1 is on.
  const domain = useMemo(() => {
    const intercepts = panels.flatMap(p => p.fitGroups.flatMap(group => fitsFor(group.stats, methods).map(fit => fit.intercept)))
    return pairDomain(points.map((p) => p.x), points.map((p) => p.y), axes, showFit ? intercepts : [])
  }, [points, panels, axes, showFit, methods])
  // Panel axes (lib/axes): panels in one group share a range. Per site gives each row of the per-season grid (one
  // site's seasons) one range; on the per-site grid Per site is Own
  const panelMode = layout === 'Combined' ? 'Shared' : axes.panels ?? 'Shared'
  const panelDomains = useMemo(() => {
    if (panelMode === 'Shared') return null
    const groups = new Map<string, typeof panels>()
    for (const p of panels) {
      const k = panelAxesKey(panelMode, p)
      groups.set(k, [...(groups.get(k) ?? []), p])
    }
    const out = new Map<string, ReturnType<typeof pairDomain>>()
    for (const ps of groups.values()) {
      const pts = ps.flatMap(p => p.points)
      const d = pairDomain(pts.map(q => q.x), pts.map(q => q.y), axes,
        showFit ? ps.flatMap(p => p.fitGroups.flatMap(group => fitsFor(group.stats, methods).map(fit => fit.intercept))) : [])
      for (const p of ps) out.set(p.key, d)
    }
    return out
  }, [panelMode, panels, axes, showFit, methods])

  const legend = useMemo(
    () =>
      (colorBy === 'Season' ? meta.seasons : colorBy === 'Group' ? (grp ? [...grp.labels.map((l, i) => ({ name: l, color: grp.colors[i] })),
        // the filters without the measurement, drawn grey: named in the legend (and hideable) when there are any
        ...(drawable.some((r) => grp.labelOf(r) === null) ? [{ name: noBand, color: NO_BAND_COLOR }] : [])] : [])
        : pmfColouring ? (meta.pmf_groups ?? []).map((g) => ({ name: g.name, color: g.color })) : meta.sites).map((s) => ({
        label: s.name,
        color: colorBy === 'Value' && !pmfColouring ? INK.neutral : s.color,
        detail: `n=${rows.filter((r) => groupOf(r) === s.name && typeof r[xField] === 'number' && typeof r[yField] === 'number').length}`,
      })),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [colorBy, pmfColouring, meta, rows, xField, yField, grp, drawable, noBand]
  )
  // Per site: the facet titles already name each site. Coloured by site the legend would list them
  // again, so it goes; coloured by local season calendars each site's seasons sit under its own title.
  const facetLegends = useMemo(() => {
    if (layout !== 'Per site' || colorBy !== 'Season' || !meta.seasons.some((s) => s.site)) return null
    return new Map(meta.sites.map((s) => {
      const own = new Set(seasonsForSite(meta, s.name).map((se) => se.name))
      return [s.name, legend.filter((it) => own.has(it.label))]
    }))
  }, [layout, colorBy, meta, legend])
  const mainLegend = layout === 'Per site' && colorBy === 'Site' ? []
    : facetLegends ? legend.filter((it) => ![...facetLegends.values()].some((items) => items.includes(it))) : legend
  const toggleLegend = useCallback((l: string) => setHidden((h) => toggleIn(h, l)), [setHidden])

  // PlotConfig.equal_axes: a 1:1 line only reads correctly on a square panel.
  const MAX_SQUARE = 560
  const fit = useFitHeight()
  const rowed = layout === 'Per season' || (layout === 'Per group' && !!grp)
  const seasonGroups = rowed ? [...new Set(panels.map((p) => p.group ?? ''))] : []
  const perGroup = rowed ? Math.max(1, ...seasonGroups.map((g) => panels.filter((p) => p.group === g).length)) : 1
  // several panels: a plot never shrinks below a readable size; when a row does not fit, it scrolls sideways
  // inside the chart (each site's seasons stay in one row, side by side) instead of squeezing every plot.
  // A Dashboard tile keeps fitting its panels to the tile.
  // a panel size picked (not Auto): every plot at that side, whatever the chart's or the window's room
  const panelPx = Math.round(Number(panelParam))
  const sized = layout !== 'Combined' && panelParam && Number.isFinite(panelPx) ? Math.min(PANEL_MAX, Math.max(PANEL_MIN, panelPx)) : 0
  const tiled = fit !== null && !sized
  const GAP = tiled ? 12 : 24
  // One phone panel fits the card; wide multi-panel rows can still scroll.
  const MIN_PLOT = sized || Math.min(tiled ? 200 : rowed ? 240 : 300, Math.max(120, width - MARGIN.left - MARGIN.right))
  const wrapSeasons = o.values.seasonRows === 'wrap'
  // wrapped: as many panels per line as fit at the minimum size
  const fitCols = Math.max(1, Math.floor((width + GAP) / (MIN_PLOT + MARGIN.left + MARGIN.right + GAP)))
  const cols = layout === 'Per site' ? (sized ? Math.min(panels.length, fitCols) : width > 700 ? 2 : 1)
    : rowed ? (tiled ? Math.min(perGroup, width > 1000 ? 4 : width > 640 ? 2 : 1) : wrapSeasons ? Math.min(perGroup, fitCols) : sized || width > 640 ? perGroup : 1)
    : 1
  const panelOuterW = Math.floor((width - (cols - 1) * GAP) / cols)
  const availW = Math.max(MIN_PLOT, panelOuterW - MARGIN.left - MARGIN.right)
  // every crossplot panel is square, with or without the 1:1 line
  const panelRows = Math.max(1, Math.ceil(panels.length / cols))
  const innerW = sized || Math.min(availW, fit === null ? (layout === 'Per site' ? 420 : rowed ? 320 : MAX_SQUARE) : Math.max(100, fit / panelRows - MARGIN.top - MARGIN.bottom))
  const innerH = innerW
  // fixed-width columns (the panel's own width): a grid wider than the chart starts at its left edge and scrolls,
  // where shrinkable columns would centre each panel in a narrower cell and clip its left side
  const gridStyle: React.CSSProperties = tiled
    ? { gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }
    : { gridTemplateColumns: `repeat(${cols}, ${innerW + MARGIN.left + MARGIN.right}px)`, columnGap: GAP, width: 'max-content', marginLeft: 'auto', marginRight: 'auto' }

  // the hover read for this chart's filters only, and each panel told only of a hover inside it
  // the hover is read inside each panel (FocusLayer), so a hover never re-renders the chart, its controls or its fits
  // the handlers read the latest state through a ref, so the panels get the same functions every render
  const live = useRef({ enter: (_e: React.MouseEvent, _pt: Point) => {}, leave: () => {}, click: (_row: FilterRow) => {}, brush: (_ids: Set<string> | null) => {} })
  live.current = {
    enter: (e: React.MouseEvent, pt: Point) => {
      const row = pt.row
      hl.setHover(row.id)
      tip.show(e, [
        row.id,
        `${row.site} · ${row.date}`,
        row.season,
        `${xField}: ${fmt(row[xField] as number)}`,
        `${yField}: ${fmt(row[yField] as number)}`,
        ...(colorBy === 'Value' ? [`${colorField}: ${fmt(pt.c)}`] : []),
        ...(sScale ? [`${sizeField}: ${fmt(pt.s)}`] : []),
        row.excluded ? `excluded: ${row.exclusion_reason}` : 'click for details',
      ])
    },
    leave: () => { hl.setHover(null); tip.hide() },
    click: (row: FilterRow) => hl.openSample(row.id),
    brush: (ids: Set<string> | null) => hl.setSelected(ids),
  }
  const stable = useMemo(() => ({
    onEnter: (e: React.MouseEvent, pt: Point) => live.current.enter(e, pt), onLeave: () => live.current.leave(),
    onClick: (row: FilterRow) => live.current.click(row), onBrush: (ids: Set<string> | null) => live.current.brush(ids),
  }), [])

  if (xField === yField) return <ChartFrame title="Scatterplot"><Empty>x and y are the same field — pick two different measurements in the bar above.</Empty></ChartFrame>

  const unitOf = (f: string) => withUnit(f, meta.field_units)


  const renderPanel = (p: (typeof panels)[number]) => (
              <ScatterPanel
                key={p.key}
                grouping={grouping}
                siteColoring={colorBy === 'Site'} seasonColoring={colorBy === 'Season'}
                title={p.title ? chartLabel(p.title) || null : null}
                titleColor={p.color}
                legend={facetLegends?.get(p.title ?? '') ?? null}
                hidden={hidden}
                hover={hover}
                onToggle={toggleLegend}
                onHover={setHover}
                points={p.points}
                stats={p.stats}
                fitGroups={p.fitGroups}
                domain={panelDomains?.get(p.key) ?? domain}
                innerW={innerW}
                innerH={innerH}
                axes={axes}
                showFit={showFit}
                xLabel={unitOf(xField)}
                yLabel={unitOf(yField)}
                colorOf={colorOf}
                radiusOf={radiusOf}
                dimOf={dimOf}
                fitDim={p.title && colorBy !== 'Season' ? dim(p.title) : 1}
                keys={p.ids}
                selected={hl.selected}
                brushing={brushing}
                onBrush={stable.onBrush}
                onEnter={stable.onEnter}
                onLeave={stable.onLeave}
                onClick={stable.onClick}
              />
  )

  return (
    <ChartFrame
      id={id}
      title={title}
      subtitle="Any measured quantity against any other, coloured by site, by the selected season calendar, or by a third measurement (the 123 'colour-coded' crossplots in the estate); a fourth can size the marks. The stats box reports R² and the regression fits selected in the sidebar. Hover anywhere near a point to find it in every other chart; click it for the full record."
      provenance="stands in for 281 scatter + 123 colour-by-third-variable figures across 116 notebooks · plotting/crossplots.py · react-graph-gallery.com/scatter-plot · /bubble-plot"
      controls={o.controls()}
      tip={domain.dropped > 0 ? `${domain.dropped} point${domain.dropped === 1 ? ' is' : 's are'} below zero and outside the displayed axes. These points are still included in the fit. Choose “Data” in the axes settings to see them.` : undefined}
    >
      <div ref={wrapRef} className={`chart-wrap centered${fit === null ? '' : ' fit-square'}`} style={fit === null ? undefined : { minHeight: fit }}>
        {points.length < 3 ? (
          <Empty>Fewer than 3 filters carry both {xField} and {yField} in this subset.</Empty>
        ) : (
          rowed && seasonGroups.length > 1
            // several sites: one group per site, its seasons side by side
            ? <div className="season-groups">{seasonGroups.map((g) => (
                <section key={g} className="season-panel-group">
                  <h4 style={{ color: meta.sites.find((x) => x.name === g)?.color }}>{g}</h4>
                  <div className="facet-grid" style={gridStyle}>
                    {panels.filter((p) => p.group === g).map(renderPanel)}
                  </div>
                </section>
              ))}</div>
            : <div className="facet-grid" style={gridStyle}>
                {panels.map(renderPanel)}
              </div>
        )}
        <div className="legend">
          {colorBy === 'Value' && cScale && !pmfColouring ? (
            <ColorLegend scale={cScale} scheme={colours.scheme} label={unitOf(colorField)} note={nNoColor > 0 ? `${nNoColor} points without ${colorField} in grey` : undefined} />
          ) : (
            <Legend items={mainLegend} hidden={hidden} onToggle={toggleLegend} onHover={setHover} highlighted={hover} note={showFit ? 'fit selection and line styles: sidebar' : undefined} />
          )}
          {sScale && <SizeLegend scale={sScale} label={unitOf(sizeField)} />}
          {sScale && nNoSize > 0 && <span className="legend-note">{nNoSize} points without {sizeField} drawn smallest</span>}
        </div>
        {colorBy === 'Value' && !pmfColouring && (
          <Legend items={legend} hidden={hidden} onToggle={(l) => setHidden((h) => toggleIn(h, l))} onHover={setHover} highlighted={hover} note="click a site to drop it; colour is the value above" />
        )}
        {layout === 'Per group' && grp && (() => {
          const n = drawable.filter((r) => grp.labelOf(r) === null).length
          return n > 0 ? <ChartAside><p className="chart-note">No data: {n.toLocaleString()} of {drawable.length.toLocaleString()} filters have no {groupingDef!.field}, so they are in no panel (colour by Group shows them in grey).</p></ChartAside> : null
        })()}
        {tip.node}
      </div>
    </ChartFrame>
  )
}

/** One axes: used once for the combined layout, once per site for the grid. */
const NO_KEYS: ReadonlySet<string> = new Set()
/** space between a readout box above the plot and the panel's edges */
const READOUT_INSET = 12

const ScatterPanel = memo(function ScatterPanel({
  title, titleColor, legend, hidden, hover, onToggle, onHover, points, stats, fitGroups, domain, innerW, innerH, axes, showFit, xLabel, yLabel, colorOf, radiusOf, dimOf, fitDim, keys, selected, brushing, onBrush, onEnter, onLeave, onClick, grouping, siteColoring, seasonColoring,
}: {
  /** the grouping the chart draws (lib/regressionPrefs effectiveGrouping), not the raw setting */
  grouping: FitGrouping
  seasonColoring: boolean
  siteColoring: boolean
  title: string | null
  titleColor: string
  /** this panel's own key (its site's seasons), under the title; null when the chart's legend covers it */
  legend: LegendItem[] | null
  hidden: Set<string>
  hover: string | null
  onToggle: (label: string) => void
  onHover: (label: string | null) => void
  points: Point[]
  stats: RegressionStats | null
  fitGroups: RegressionGroup[]
  domain: PairDomain
  innerW: number
  innerH: number
  axes: AxesOpts
  showFit: boolean
  xLabel: string
  yLabel: string
  colorOf: (p: Point) => string
  radiusOf: (p: Point) => number
  dimOf: (p: Point) => number
  fitDim: number
  /** the panel's filter ids: its FocusLayer listens to a hover or pin of these only */
  keys: ReadonlySet<string>
  selected: Set<string> | null
  brushing: boolean
  onBrush: (ids: Set<string> | null) => void
  onEnter: (e: React.MouseEvent, p: Point) => void
  onLeave: () => void
  onClick: (row: FilterRow) => void
}) {
  const clipId = useId()
  const edits = useFilterEdits()
  const mk = (dom: [number, number], range: [number, number]): Scale => {
    if (axes.log) return d3.scaleLog().domain([Math.max(1e-4, dom[0]), dom[1]]).range(range).nice()
    const scale = d3.scaleLinear().domain(dom).range(range)
    // Keep the compact intercept margins and equal spans from pairDomain.
    return axes.mode === 'Show intercept' ? scale : scale.nice()
  }
  // the same scales until the axes change, so a hover (which re-renders the panel) keeps the Delaunay index below
  const { xScale, yScale } = useMemo(() => ({ xScale: mk(domain.x, [0, innerW]), yScale: mk(domain.y, [innerH, 0]) }),
    [domain, innerW, innerH, axes.log, axes.mode]) // eslint-disable-line react-hooks/exhaustive-deps
  const [d0, d1] = xScale.domain() as [number, number]
  // On log axes a fitted line goes negative somewhere left of the data, and a
  // log scale maps that to NaN (React then drops the attribute). Clamp the
  // segment's y to the bottom of the domain; the clipPath hides the rest.
  const yFloor = yScale.domain()[0] as number
  const yv = (v: number) => (axes.log ? Math.max(yFloor, v) : v)
  const drawn = useMemo(() => points.filter((p) => inDomain(domain, p.x, p.y)), [points, domain])
  const dots = useMemo(() => drawn.map((p) => {
    const st = focusStyle(p.row.id, null, { r: radiusOf(p) }, selected)
    const k = dimOf(p)
    return (
      <circle
        key={p.row.id}
        data-export-season={seasonColoring ? p.row.season : undefined} data-export-site={siteColoring ? p.row.site : undefined} data-export-color={siteColoring || seasonColoring ? colorOf(p) : undefined}
        cx={xScale(p.x)}
        cy={yScale(p.y)}
        r={st.r}
        fill={colorOf(p)}
        fillOpacity={st.opacity * k}
        stroke={st.stroke}
        strokeOpacity={k}
        strokeWidth={st.strokeWidth}
        data-dot={k === 1 ? st.dot : undefined}
        pointerEvents="none"
      />
    )
  }), [drawn, xScale, yScale, selected, colorOf, radiusOf, dimOf, siteColoring, seasonColoring])
  // closest-point detection: one Delaunay over the drawn points, one hit rect
  const delaunay = useMemo(() => d3.Delaunay.from(drawn, (p) => xScale(p.x), (p) => yScale(p.y)), [drawn, xScale, yScale])
  const hoverRef = useRef<Point | null>(null)
  const dotsRef = useRef<SVGGElement>(null)
  const HIT = 18
  // d3.brush owns its own DOM, so it is attached imperatively to an empty <g>
  const brushRef = useRef<SVGGElement>(null)
  const brushObj = useRef<d3.BrushBehavior<unknown> | null>(null)
  useEffect(() => {
    const g = brushRef.current
    if (!g) return
    const sel = d3.select(g)
    if (!brushing) { sel.selectAll('*').remove(); sel.on('.brush', null); return }
    const brush = d3.brush<unknown>()
      .extent([[0, 0], [innerW, innerH]])
      .on('end', (ev: d3.D3BrushEvent<unknown>) => {
        // the box is cleared below once the points are picked; that clear is not the reader's
        if (!ev.sourceEvent) return
        if (!ev.selection) { onBrush(null); return }
        const [[x0, y0], [x1, y1]] = ev.selection as [[number, number], [number, number]]
        const ids = new Set<string>()
        for (const p of drawn) {
          const px = xScale(p.x)
          const py = yScale(p.y)
          if (px >= x0 && px <= x1 && py >= y0 && py <= y1) ids.add(p.row.id)
        }
        onBrush(ids.size ? ids : null)
        // keep the picked points, not the rectangle: they stay highlighted on every chart
        sel.call(brush.move, null)
      })
    sel.call(brush)
    styleBrush(sel)
    brushObj.current = brush
    return () => { sel.on('.brush', null); sel.selectAll('*').remove() }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [brushing, drawn, innerW, innerH])
  // a selection cleared elsewhere (the brush bar's keep / exclude / ×) takes the drawn box with it
  useEffect(() => {
    if (!selected && brushRef.current && brushObj.current && d3.brushSelection(brushRef.current)) d3.select(brushRef.current).call(brushObj.current.move, null)
  }, [selected])
  const nearest = (e: React.MouseEvent<SVGRectElement>): Point | null => {
    if (drawn.length === 0) return null
    const [mx, my] = d3.pointer(e)
    const i = delaunay.find(mx, my)
    const p = drawn[i]
    if (!p) return null
    const dx = xScale(p.x) - mx
    const dy = yScale(p.y) - my
    return Math.hypot(dx, dy) <= HIT + radiusOf(p) ? p : null
  }
  const onMove = (e: React.MouseEvent<SVGRectElement>) => {
    const p = nearest(e)
    if (p !== hoverRef.current) {
      hoverRef.current = p
      if (p) onEnter(e, p)
      else onLeave()
    } else if (p) onEnter(e, p)
  }
  const markIntercepts = axes.mode === 'Show intercept' && showFit && !axes.log && stats
  const originAxes = axes.mode === 'Show intercept' && !axes.log && xScale.domain()[0] < 0 && yScale.domain()[0] < 0
  const { methods } = useRegressions()
  const siteFits = grouping !== 'pooled' && showFit && methods.length > 0
  const selectedFits = fitGroups.flatMap(group => fitsFor(group.stats, showFit ? methods : []).map(fit => ({ ...fit, color: group.color ?? fit.color, name: group.name })))
  const svgWidth = innerW + MARGIN.left + MARGIN.right
  const summary = useRegressionSummary({ stats, groups: siteFits ? fitGroups : undefined, width: svgWidth - 2 * READOUT_INSET, enabled: showFit,
    panel: title ? { title, color: siteColoring ? titleColor : undefined } : null })
  const top = MARGIN.top + summary.height

  return (
    <div>
      {title && (
        <p className="facet-title" data-export-site={title ?? undefined} data-export-color={titleColor}>
          {/* the site's colour only means something when the points carry it */}
          {siteColoring && <span className="swatch" style={{ background: titleColor }} />} {title}
          <span style={{ color: 'var(--ink-muted)', fontWeight: 400, fontFamily: 'var(--mono)', fontSize: 11 }}>n={points.length}</span>
        </p>
      )}
      {legend && legend.length > 0 && (
        <ChartSite name={title}>
          <div className="facet-legend">
            <Legend items={legend} hidden={hidden} onToggle={onToggle} onHover={onHover} highlighted={hover} />
          </div>
        </ChartSite>
      )}
      <svg width={svgWidth} height={innerH + top + MARGIN.bottom} className="animated">
        <defs>
          {/* fit and identity lines are clipped to the axes: a slope above 1 otherwise leaves through the top margin */}
          <clipPath id={clipId}>
            <rect x={0} y={0} width={innerW} height={innerH} />
          </clipPath>
        </defs>
        <g transform={`translate(${READOUT_INSET},6)`}>{summary.node}</g>
        <g className="regression-plot" transform={`translate(${MARGIN.left},${top})`}>
          {/* In 'Show intercept' mode the spines cross at the origin, matplotlib-style, so the
              intercept markers sit on the y axis and the x axis runs through the cloud. */}
          {originAxes ? (
            <>
              <YAxis scale={yScale} x={xScale(0)} label={yLabel} gridWidth={innerW} gridFrom={0} labelX={-50} pick="y" />
              <XAxis scale={xScale} y={yScale(0)} label={xLabel} labelDy={innerH - yScale(0) + 40} pick="x" />
            </>
          ) : (
            <>
              <YAxis scale={yScale} x={0} label={yLabel} gridWidth={innerW} pick="y" />
              <XAxis scale={xScale} y={innerH} label={xLabel} pick="x" />
            </>
          )}

          <g clipPath={`url(#${clipId})`}>
            {axes.identity && (
              <IdentityLine x1={xScale(d0)} y1={yScale(d0)} x2={xScale(d1)} y2={yScale(d1)} />
            )}
            {stats && showFit && (
              <g opacity={fitDim}>
                {grouping !== 'pooled' ? <RegressionGroupLines groups={fitGroups} x={xScale} y={v => yScale(yv(v))} width={innerW} height={innerH} />
                  : <RegressionLines stats={stats} x={xScale} y={v => yScale(yv(v))} domain={xScale.domain()} width={innerW} height={innerH} />}
              </g>
            )}
          </g>

          {/* the dots do not depend on the hover: a focus dims them all with one class (styles.css .scatter-dots)
              and draws the focused filter again on top, so a hover re-renders two circles, not every dot */}
          <g ref={dotsRef} className="scatter-dots">{dots}</g>
          <FocusLayer keys={keys} items={drawn} idOf={(p) => p.row.id} groupRef={dotsRef} render={(p) => {
            const st = focusStyle(p.row.id, p.row.id, { r: radiusOf(p) })
            return <circle cx={xScale(p.x)} cy={yScale(p.y)} r={st.r} fill={colorOf(p)} fillOpacity={st.opacity * dimOf(p)}
              stroke={st.stroke} strokeWidth={st.strokeWidth} pointerEvents="none" />
          }} />

          {/* the hit layer: closest point wins, within HIT px of the pointer; the brush replaces it while selecting */}
          {!brushing && (
            <rect
              x={0} y={0} width={innerW} height={innerH} fill="transparent"
              style={{ cursor: hoverRef.current ? 'pointer' : 'default' }}
              onMouseMove={onMove}
              onMouseLeave={() => { hoverRef.current = null; onLeave() }}
              onClick={(e) => {
                const p = nearest(e)
                if (p) { e.stopPropagation(); onClick(p.row) }
                else { hoverRef.current = null; onLeave() }
              }}
              onContextMenu={(e) => {
                // the nearest point gets the filter menu; on empty plot the page-level menu offers the brush and undo
                const p = nearest(e)
                if (p) edits.openMenu(e, { ids: [p.row.id], title: p.row.id })
              }}
            />
          )}
          <g ref={brushRef} className="brush-layer" />

          {markIntercepts && (
            <g pointerEvents="none" fontFamily={FONT.mono} fontSize={10.5}>
              {!originAxes && <RefLine orient="v" at={xScale(0)} from={0} to={innerH} />}
              {selectedFits.map(fit => <g key={fit.name + fit.method}>
                <circle cx={xScale(0)} cy={yScale(fit.intercept)} r={4.5} fill={INK.panel} stroke={fit.color} data-regression-stroke={fit.method} strokeWidth={2} />
                <text x={xScale(0) + 8} y={yScale(fit.intercept)} dy="0.35em" fill={fit.color} data-regression-color={fit.method} stroke={INK.panel} strokeWidth={3} paintOrder="stroke">{fit.label} b = {fmt(fit.intercept, 3)}</text>
              </g>)}
            </g>
          )}

        </g>
      </svg>
    </div>
  )
})

