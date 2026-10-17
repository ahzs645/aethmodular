import { fitUncertainty } from '@/lib/fitUncertainty'
import { styleBrush } from '@/lib/brush'
import { seasonsForSite } from '@/siteSeasons'
import { IdentityLine, RefLine } from '@/components/RefLines'
import { RegressionLines, RegressionReadout, RegressionGroupLines, RegressionGroupReadouts, regressionGroupReadoutLayout } from '@/components/RegressionOverlay'
import { useRegressions, fitsFor } from '@/lib/regressionPrefs'
import { fitGroupingOption, useChartOptions } from '@/lib/chartOptions'
import { groupedRegressions, type FitGrouping, type RegressionGroup } from '@/lib/siteRegression'
import { useEffect, useId, useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Note } from '@/components/ChartFrame'
import { ColorLegend, SizeLegend } from '@/components/ColorLegend'
import { FieldSelect } from '@/components/FieldSelect'
import { Legend, toggleIn, useLegend } from '@/components/Legend'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { focusStyle, useHighlight } from '@/lib/highlight'
import { useFilterEdits } from '@/lib/filterEdits'
import { inDomain, pairDomain, type AxesOpts, type PairDomain, robustDomain } from '@/lib/axes'
import { regression, fmt, type RegressionStats } from '@/lib/stats'
import { INK, MARGIN, FONT, RAMP_SEQUENTIAL, withUnit } from '@/lib/theme'
import type { FilterRow, MetaFile } from '@/lib/types'
import { NO_PMF, PMF_SOURCE } from '@/lib/pmfGroups'

type Point = { row: FilterRow; x: number; y: number; c: number | null; s: number | null }
type Scale = d3.ScaleLinear<number, number> | d3.ScaleLogarithmic<number, number>

/** Per season: one site's seasons side by side, or each site's seasons grouped under it */
const LAYOUTS = ['Combined', 'Per site', 'Per season'] as const
const COLOR_BY = ['Site', 'Season', 'Value'] as const
/** Value's categorical choice where a site in view has PMF: each point by its day's dominant source */
const PMF_COLOUR = 'PMF source (dominant)'
const NO_SIZE = 'none'

/** Sequential ramp over the 2–98 % bulk of a third variable, so one outlier does not wash the rest to the first colour. */
function valueScale(values: number[]) {
  const [lo, hi] = robustDomain(values, 0.02, 0.98).domain
  return d3.scaleSequential(d3.interpolateRgbBasis(RAMP_SEQUENTIAL.slice(2))).domain([lo, hi === lo ? lo + 1 : hi]).clamp(true)
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
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const tip = useTooltip(wrapRef)
  const hl = useHighlight()
  const { methods, grouping: chosenGrouping, setGrouping } = useRegressions()
  const { hidden, setHidden, dim, hover, setHover } = useLegend()
  // the reader's colour field, or Iron, or the first field, among the fields this view has now: the
  // chart can mount while a network's fields are still loading (empty), or keep a field the next network lacks
  const [colorChoice, setColorField] = useState<string | null>(null)
  const hasPmf = !!meta.pmf_groups?.length && rows.some((r) => typeof r[PMF_SOURCE] === 'string')
  const colorField = [colorChoice, 'Iron', meta.fields[0]].find((f): f is string => !!f && (meta.fields.includes(f) || (f === PMF_COLOUR && hasPmf))) ?? ''
  const [sizeField, setSizeField] = useState(NO_SIZE)
  // the options and when each applies (lib/chartOptions). Fits are per panel, so the grouping
  // offered follows the layout: a fit per site inside a site's panel is the pooled fit, and a
  // per-season panel holds one site in one season, so every grouping is its one fit
  const o = useChartOptions('scatter', [
    { key: 'layout', label: 'Layout', choices: LAYOUTS, default: 'Combined', title: 'PlotConfig.layout: combined overlays sites; per site is the grid' },
    { key: 'colorBy', label: 'Colour', choices: COLOR_BY, default: 'Site', title: 'Value: a third measurement on a sequential ramp (2–98 % of its range), or, where a site in view has PMF, its day\'s dominant source', onChange: () => setHidden(new Set()) },
    // where a site in view has PMF (lib/pmfGroups; Addis so far), its dominant source heads the list, beside the PMF shares
    { key: 'colorField', when: (c) => c.colorBy === 'Value', node: () => <FieldSelect label="by" value={colorField} meta={meta} onChange={(f) => { setColorField(f); setHidden(new Set()) }} extraOptions={hasPmf ? [PMF_COLOUR] : []} /> },
    { key: 'size', node: () => <FieldSelect label="Size" value={sizeField} meta={meta} onChange={setSizeField} extraOptions={[NO_SIZE]} title="Bubble plot: mark area proportional to a fourth measurement" /> },
    { key: 'fitLines', label: 'Fit lines', toggle: true, default: true },
    fitGroupingOption({ bind: [chosenGrouping, setGrouping as (v: never) => void],
      when: (c) => !!c.fitLines && (c.estimators as number) > 0 && c.layout !== 'Per season',
      hiddenNote: (c) => (c.fitLines && (c.estimators as number) > 0 && c.layout === 'Per season' ? 'one fit per panel' : null),
      offerSite: (c) => c.layout === 'Combined',
      pooledLabel: (c) => (c.layout === 'Per site' ? 'One fit per site' : 'One pooled fit') }),
    { key: 'brushing', label: 'Brush to select', toggle: true, default: false, title: 'Drag a rectangle: the enclosed filters are highlighted on every chart of this tab', onChange: ((v: boolean) => { if (!v) hl.setSelected(null) }) as (v: never) => void },
    { key: 'selected', when: () => !!hl.selected, node: () => <span className="control"><strong>{hl.selected?.size}</strong> selected<button type="button" className="btn quiet" onClick={() => hl.setSelected(null)}>Clear</button></span> },
    { key: 'count', node: () => <span className="control">{points.length} points</span> },
  ], { estimators: methods.length })
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
  const groupOf = (r: FilterRow) => (colorBy === 'Season' ? r.season : pmfColouring ? pmfOf(r) : r.site)
  const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)

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
    [rows, xField, yField, logAxes, hidden, colorBy, colorField, sizeField]
  )

  // third variable → colour; fourth → size (sqrt, so area is proportional)
  const cScale = useMemo(() => (colorBy === 'Value' && !pmfColouring ? valueScale(points.map((p) => p.c).filter((v): v is number => v !== null)) : null), [points, colorBy, pmfColouring])
  const sScale = useMemo(() => {
    if (sizeField === NO_SIZE) return null
    const vals = points.map((p) => p.s).filter((v): v is number => v !== null && v > 0)
    return d3.scaleSqrt().domain([0, d3.quantile(vals.slice().sort(d3.ascending), 0.98) ?? 1]).range([1.5, 9]).clamp(true)
  }, [points, sizeField])
  const nNoColor = colorBy === 'Value' ? points.filter((p) => p.c === null).length : 0
  const nNoSize = sScale ? points.filter((p) => p.s === null).length : 0

  const colorOf = (p: Point) =>
    pmfColouring ? pmfColor.get(pmfOf(p.row)) ?? INK.neutral
      : colorBy === 'Value' ? (p.c === null || !cScale ? INK.neutral : cScale(p.c)) : colorBy === 'Season' ? seasonColor.get(p.row.season) ?? INK.muted
      : (p.row.color as string)
  const radiusOf = (p: Point) => (sScale ? (p.s === null ? 1.5 : sScale(p.s)) : 4)

  const panels = useMemo(() => {
    const raw: { key: string; title: string | null; color: string; points: Point[]; group?: string }[] = layout === 'Combined'
      ? [{ key: 'all', title: null as string | null, color: INK.text, points }]
      : layout === 'Per season'
        // each site's own seasons under the active calendar (local calendars carry the site in the name)
        ? meta.sites.flatMap((s) => seasonsForSite(meta, s.name).map((se) => ({
            key: `${s.code}|${se.name}`, group: s.name, color: se.color,
            title: se.site && se.name.startsWith(`${se.site} · `) ? se.name.slice(se.site.length + 3) : se.name,
            points: points.filter((p) => p.row.site === s.name && p.row.season === se.name),
          }))).filter((p) => p.points.length > 0)
        : meta.sites
          .map((s) => ({ key: s.code, title: s.name, color: s.color, points: points.filter((p) => p.row.site === s.name) }))
          .filter((p) => p.points.length > 0)
    // Deming lambda: when both axes are the same measured quantity we default to
    // 1.0 (orthogonal), exactly as calculate_regression_stats does.
    return raw.map((p) => ({
      ...p,
      stats: regression(p.points.map((q) => q.x), p.points.map((q) => q.y), fitUncertainty(p.points.map(q => q.row), xField, yField, meta)),
      fitGroups: groupedRegressions(p.points, grouping, { x: q => q.x, y: q => q.y, site: q => q.row.site, season: q => q.row.season, seasonColor: q => meta.seasons.find(se => se.name === q.row.season)?.color ?? INK.muted,
        color: q => meta.sites.find(s => s.name === q.row.site)?.color ?? INK.muted,
        options: members => fitUncertainty(members.map(q => q.row), xField, yField, meta) }),
    }))
  }, [layout, points, meta, xField, yField, grouping])

  // Shared domain across every panel by default — a per-site grid with different axes
  // per panel silently hides that one site's slope is twice another's. "Own" panel axes
  // (sidebar) fit each site's panel to its own points, still square when 1:1 is on.
  const domain = useMemo(() => {
    const intercepts = panels.flatMap(p => p.fitGroups.flatMap(group => fitsFor(group.stats, methods).map(fit => fit.intercept)))
    return pairDomain(points.map((p) => p.x), points.map((p) => p.y), axes, showFit ? intercepts : [])
  }, [points, panels, axes, showFit, methods])
  // 'Own' panel axes apply to any split layout (one panel per site, or per site and season)
  const ownAxes = layout !== 'Combined' && axes.panels === 'Own'
  const panelDomains = useMemo(() => ownAxes
    ? new Map(panels.map(p => [p.key, pairDomain(p.points.map(q => q.x), p.points.map(q => q.y), axes,
      showFit ? p.fitGroups.flatMap(group => fitsFor(group.stats, methods).map(fit => fit.intercept)) : [])]))
    : null, [ownAxes, panels, axes, showFit, methods])

  const legend = useMemo(
    () =>
      (colorBy === 'Season' ? meta.seasons : pmfColouring ? (meta.pmf_groups ?? []).map((g) => ({ name: g.name, color: g.color })) : meta.sites).map((s) => ({
        label: s.name,
        color: colorBy === 'Value' && !pmfColouring ? INK.neutral : s.color,
        detail: `n=${rows.filter((r) => groupOf(r) === s.name && typeof r[xField] === 'number' && typeof r[yField] === 'number').length}`,
      })),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [colorBy, pmfColouring, meta, rows, xField, yField]
  )

  // PlotConfig.equal_axes: a 1:1 line only reads correctly on a square panel.
  const MAX_SQUARE = 560
  const seasonGroups = layout === 'Per season' ? [...new Set(panels.map((p) => p.group ?? ''))] : []
  const perGroup = layout === 'Per season' ? Math.max(1, ...seasonGroups.map((g) => panels.filter((p) => p.group === g).length)) : 1
  const cols = layout === 'Per site' && width > 700 ? 2 : layout === 'Per season' ? Math.min(perGroup, width > 1000 ? 4 : width > 640 ? 2 : 1) : 1
  const panelOuterW = Math.floor((width - (cols - 1) * 12) / cols)
  const availW = Math.max(200, panelOuterW - MARGIN.left - MARGIN.right)
  // every crossplot panel is square, with or without the 1:1 line
  const innerW = Math.min(availW, layout === 'Per site' ? 420 : layout === 'Per season' ? 320 : MAX_SQUARE)
  const innerH = innerW

  if (xField === yField) return <ChartFrame title="Scatterplot"><Empty>x and y are the same field — pick two different measurements in the bar above.</Empty></ChartFrame>

  const unitOf = (f: string) => withUnit(f, meta.field_units)

  const renderPanel = (p: (typeof panels)[number]) => (
              <ScatterPanel
                key={p.key}
                grouping={grouping}
                title={p.title}
                titleColor={p.color}
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
                dimOf={(pt) => dim(groupOf(pt.row))}
                fitDim={p.title && colorBy !== 'Season' ? dim(p.title) : 1}
                focusId={hl.focusId}
                selected={hl.selected}
                brushing={brushing}
                onBrush={(ids) => hl.setSelected(ids)}
                onEnter={(e, pt) => {
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
                }}
                onLeave={() => { hl.setHover(null); tip.hide() }}
                onClick={(row) => hl.openSample(row.id)}
              />
  )

  return (
    <ChartFrame
      id={id}
      title={title}
      subtitle="Any measured quantity against any other, coloured by site, by the selected season calendar, or by a third measurement (the 123 'colour-coded' crossplots in the estate); a fourth can size the marks. The stats box reports R² and the regression fits selected in the sidebar. Hover anywhere near a point to find it in every other chart; click it for the full record."
      provenance="stands in for 281 scatter + 123 colour-by-third-variable figures across 116 notebooks · plotting/crossplots.py · react-graph-gallery.com/scatter-plot · /bubble-plot"
      controls={o.controls()}
    >
      <div ref={wrapRef} className="chart-wrap centered">
        {points.length < 3 ? (
          <Empty>Fewer than 3 filters carry both {xField} and {yField} in this subset.</Empty>
        ) : (
          layout === 'Per season' && seasonGroups.length > 1
            // several sites: one group per site, its seasons side by side
            ? <div className="season-groups">{seasonGroups.map((g) => (
                <section key={g} className="season-panel-group">
                  <h4 style={{ color: meta.sites.find((x) => x.name === g)?.color }}>{g}</h4>
                  <div className="facet-grid" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>
                    {panels.filter((p) => p.group === g).map(renderPanel)}
                  </div>
                </section>
              ))}</div>
            : <div className="facet-grid" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>
                {panels.map(renderPanel)}
              </div>
        )}
        <div className="legend">
          {colorBy === 'Value' && cScale && !pmfColouring ? (
            <ColorLegend scale={cScale} label={unitOf(colorField)} note={nNoColor > 0 ? `${nNoColor} points without ${colorField} in grey` : undefined} />
          ) : (
            <Legend items={legend} hidden={hidden} onToggle={(l) => setHidden((h) => toggleIn(h, l))} onHover={setHover} highlighted={hover} note={showFit ? 'fit selection and line styles: sidebar' : undefined} />
          )}
          {sScale && <SizeLegend scale={sScale} label={unitOf(sizeField)} />}
          {sScale && nNoSize > 0 && <span className="legend-note">{nNoSize} points without {sizeField} drawn smallest</span>}
        </div>
        {colorBy === 'Value' && !pmfColouring && (
          <Legend items={legend} hidden={hidden} onToggle={(l) => setHidden((h) => toggleIn(h, l))} onHover={setHover} highlighted={hover} note="click a site to drop it; colour is the value above" />
        )}
        {domain.dropped > 0 && (
          <Note>{domain.dropped} point{domain.dropped === 1 ? '' : 's'} below zero are outside the axes and not drawn; they are still in the fit. Switch axes to “Data” in the bar to see them.</Note>
        )}
        {tip.node}
      </div>
    </ChartFrame>
  )
}

/** One axes: used once for the combined layout, once per site for the grid. */
function ScatterPanel({
  title, titleColor, points, stats, fitGroups, domain, innerW, innerH, axes, showFit, xLabel, yLabel, colorOf, radiusOf, dimOf, fitDim, focusId, selected, brushing, onBrush, onEnter, onLeave, onClick, grouping,
}: {
  /** the grouping the chart draws (lib/regressionPrefs effectiveGrouping), not the raw setting */
  grouping: FitGrouping
  title: string | null
  titleColor: string
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
  focusId: string | null
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
  const xScale = mk(domain.x, [0, innerW])
  const yScale = mk(domain.y, [innerH, 0])
  const [d0, d1] = xScale.domain() as [number, number]
  // On log axes a fitted line goes negative somewhere left of the data, and a
  // log scale maps that to NaN (React then drops the attribute). Clamp the
  // segment's y to the bottom of the domain; the clipPath hides the rest.
  const yFloor = yScale.domain()[0] as number
  const yv = (v: number) => (axes.log ? Math.max(yFloor, v) : v)
  const drawn = useMemo(() => points.filter((p) => inDomain(domain, p.x, p.y)), [points, domain])
  // closest-point detection: one Delaunay over the drawn points, one hit rect
  const delaunay = useMemo(() => d3.Delaunay.from(drawn, (p) => xScale(p.x), (p) => yScale(p.y)), [drawn, xScale, yScale])
  const hoverRef = useRef<Point | null>(null)
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
        if (!ev.selection) { onBrush(null); return }
        const [[x0, y0], [x1, y1]] = ev.selection as [[number, number], [number, number]]
        const ids = new Set<string>()
        for (const p of drawn) {
          const px = xScale(p.x)
          const py = yScale(p.y)
          if (px >= x0 && px <= x1 && py >= y0 && py <= y1) ids.add(p.row.id)
        }
        onBrush(ids.size ? ids : null)
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
  const { methods, showCI } = useRegressions()
  const siteFits = grouping !== 'pooled' && showFit && methods.length > 0
  const selectedFits = fitGroups.flatMap(group => fitsFor(group.stats, showFit ? methods : []).map(fit => ({ ...fit, color: group.color ?? fit.color, name: group.name })))
  const svgWidth = innerW + MARGIN.left + MARGIN.right
  const readoutHeight = siteFits ? regressionGroupReadoutLayout(fitGroups.length, svgWidth, methods, showCI).height : 0

  return (
    <div>
      {title && (
        <p className="facet-title">
          <span className="swatch" style={{ background: titleColor }} /> {title}
          <span style={{ color: 'var(--ink-muted)', fontWeight: 400, fontFamily: 'var(--mono)', fontSize: 11 }}>n={points.length}</span>
        </p>
      )}
      <svg width={svgWidth} height={innerH + MARGIN.top + MARGIN.bottom + readoutHeight} className="animated">
        <defs>
          {/* fit and identity lines are clipped to the axes: a slope above 1 otherwise leaves through the top margin */}
          <clipPath id={clipId}>
            <rect x={0} y={0} width={innerW} height={innerH} />
          </clipPath>
        </defs>
        <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
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

          {drawn.map((p) => {
            const st = focusStyle(p.row.id, focusId, { r: radiusOf(p), opacity: 0.72 }, selected)
            const k = dimOf(p)
            return (
              <circle
                key={p.row.id}
                cx={xScale(p.x)}
                cy={yScale(p.y)}
                r={st.r}
                fill={colorOf(p)}
                fillOpacity={st.opacity * k}
                stroke={st.stroke}
                strokeOpacity={k}
                strokeWidth={st.strokeWidth}
                pointerEvents="none"
              />
            )
          })}

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
                <circle cx={xScale(0)} cy={yScale(fit.intercept)} r={4.5} fill="#fff" stroke={fit.color} strokeWidth={2} />
                <text x={xScale(0) + 8} y={yScale(fit.intercept)} dy="0.35em" fill={fit.color} stroke="#fff" strokeWidth={3} paintOrder="stroke">{fit.label} b = {fmt(fit.intercept, 3)}</text>
              </g>)}
            </g>
          )}

          {grouping === 'pooled' && showFit && <RegressionReadout stats={stats} />}
        </g>
        {siteFits && <RegressionGroupReadouts groups={fitGroups} width={svgWidth} y={innerH + MARGIN.top + MARGIN.bottom} />}
      </svg>
    </div>
  )
}
