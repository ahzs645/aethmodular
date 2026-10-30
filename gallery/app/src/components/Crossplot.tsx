import { useEffect, useId, useMemo, useRef, type ReactNode } from 'react'
import { IdentityLine } from '@/components/RefLines'
import * as d3 from 'd3'
import { RegressionLines, RegressionBand, RegressionGroupLines, useRegressionSummary } from '@/components/RegressionOverlay'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { groupedRegressions, type FitGrouping } from '@/lib/siteRegression'
import { fidProps, useFilterEdits, useHandEdits } from '@/lib/filterEdits'
import { monthBlocks } from '@/lib/regressionCI'
import { regression, type RegressionStats } from '@/lib/stats'
import { useRegressions, FitGroupingControl } from '@/lib/regressionPrefs'
import type { Estimator } from '@/lib/regressionMethods'
import { baseFilterId, dotFill, dotRim, FocusLayer, PinRing, useHighlight, type FilterRecord } from '@/lib/highlight'
import { HitLayer } from '@/components/HitLayer'
import { FONT, INK } from '@/lib/theme'
import { Toggle } from '@/components/ChartFrame'
import { styleBrush } from '@/lib/brush'
import { useTileSetting } from '@/lib/url'
import { ChartAside } from '@/lib/singleView'

/** One filter on a crossplot. `rec` opens the record drawer; `id` links hover and pin across charts. */
export interface CrossPoint {
  x: number; y: number; color: string; hollow?: boolean; tip: string[]; id?: string; rec?: FilterRecord
  /** opacity multiplier, e.g. a legend dimming every series but the hovered one */
  opacity?: number
  /** for a fit per site or per season (else the record's site; no season, no per-season fit) */
  site?: string; season?: string; seasonColor?: string
  /** what a click opens (a gallery filter's record in the sample drawer); else `rec` in the record drawer */
  open?: () => void
}

/** A fixed line drawn over the points (a saved fit, a season's line) with its confidence band. */
export interface CrossLine { slope: number; intercept: number; color: string; label: string; dash?: string; method: Estimator; stats: RegressionStats | null }

/**
 * The research pages' 1:1 crossplot: a square plot area with identical x and y
 * ranges, so the dashed line is the true diagonal. Every point is shown (the
 * research pages keep the "Data" axes, docs/gallery-chart-standards.md §5).
 * Hover and click link the filter across every chart and open its record.
 * With `fit`, the browser refits the drawn points under the chosen estimators
 * (pooled, or one line per site); without it the plot only shows saved results.
 */
/** a crossplot axis range: 0 always in view, 5 % headroom, nice ends */
const span = (v: number[]) => d3.scaleLinear().domain([Math.min(0, d3.min(v) ?? 0), (d3.max(v) ?? 1) * 1.05]).nice().domain() as [number, number]

/**
 * One range for several panels (Panel axes: Shared, lib/panelAxes): pass the result's `domain` and
 * `yDomain` to every panel. With `sameScale` both axes share one span, as a single panel's would.
 */
export function sharedCrossDomains(sets: CrossPoint[][], sameScale = true): { domain: [number, number]; yDomain?: [number, number] } {
  const pts = sets.flat().filter((p) => p.x != null && p.y != null && Number.isFinite(p.x) && Number.isFinite(p.y))
  const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y)
  return sameScale ? { domain: span([...xs, ...ys]) } : { domain: span(xs), yDomain: span(ys) }
}

const NO_KEYS: ReadonlySet<string> = new Set()

export function Crossplot({ points, title, xLabel, yLabel, maxSide = 360, domain, fit = true, errorRatio, extraLines = [], grouping: groupingOverride, groupNoun, background, pick, sameScale = true, yDomain, brushable = false, brush }: {
  points: CrossPoint[]
  /** Offer rectangular selection with the shared filter review and hand-edit actions. */
  brushable?: boolean
  /** the brush toggle held by the chart (useCrossplotBrush), whose <CrossplotBrushControls> sit in its options row; else the plot draws its own above it */
  brush?: boolean
  title: string
  xLabel: string
  yLabel: string
  maxSide?: number
  /** a fixed [lo, hi] shared by both axes, e.g. so two panels compare directly; else from the points */
  domain?: [number, number]
  /** refit the drawn points (Deming with `errorRatio`, OLS, York as chosen) */
  fit?: boolean
  /** Deming error-variance ratio λ for the refit */
  errorRatio?: number | null
  extraLines?: CrossLine[]
  /** the chart's own fit grouping (with its own control), instead of the gallery-wide setting; 'none' draws no fits */
  grouping?: FitGrouping | 'none'
  /** what one fitted population is called in the note under a per-group fit (default its site or season) */
  groupNoun?: string
  /** drawn under the points on the same scales, e.g. a density of the wider population */
  background?: (x: d3.ScaleLinear<number, number>, y: d3.ScaleLinear<number, number>) => ReactNode
  /** axis titles open the nearest axis picker (components/AxisFieldPopover: the page's or the gallery's); 'x' or 'y' for one */
  pick?: boolean | 'x' | 'y'
  /** one range and a 1:1 line for both axes (comparable measurements); false: each axis its own range */
  sameScale?: boolean
  /** with sameScale false, a fixed y range (e.g. shared with a neighbouring panel); `domain` is then x's */
  yDomain?: [number, number]
}) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  // pins and handlers without re-rendering on hovers; the hover is read below for this panel's filters only
  const hl = useHighlight(NO_KEYS)
  const m = { top: 24, right: 14, bottom: brushable ? 62 : 46, left: 56 }
  const side = Math.max(160, Math.min(width - m.left - m.right, maxSide))
  const iw = side, ih = side, w = side + m.left + m.right
  const clipId = useId()
  // a filter excluded by hand on any page is left out here too, points and refit alike
  const edits = useHandEdits()
  const menu = useFilterEdits()
  const [ownBrush, setOwnBrush] = useCrossplotBrush()
  const brushing = brushable && (brush ?? ownBrush)
  const brushRef = useRef<SVGGElement>(null)
  const drawable = useMemo(() => points.filter(p => p.x != null && p.y != null && Number.isFinite(p.x) && Number.isFinite(p.y)), [points])
  const valid = useMemo(() => drawable.filter(p => !edits.isOut(p.id ?? p.rec?.id)), [drawable, edits])
  const handNote = edits.note(drawable.length - valid.length, { refit: fit && groupingOverride !== 'none', saved: extraLines.length > 0 })
  const lambda = errorRatio ?? 1
  const prefs = useRegressions()
  const { methods } = prefs
  const fitting = fit && groupingOverride !== 'none'
  const grouping: FitGrouping = groupingOverride && groupingOverride !== 'none' ? groupingOverride : prefs.grouping
  const siteOf = (p: CrossPoint) => p.site ?? p.rec?.site ?? 'Unknown site'
  const multiSite = fitting && new Set(valid.map(siteOf)).size > 1
  // a fit per season only where the points say their season
  const seasons = fitting && valid.some((p) => p.season)
  const noun = groupNoun ?? (grouping === 'season' ? 'season' : 'site')
  const siteFits = methods.length > 0 && ((grouping === 'site' && multiSite) || (grouping === 'season' && seasons))
  const fitGroups = useMemo(() => fitting ? groupedRegressions(valid, grouping, {
    x: (p) => p.x, y: (p) => p.y, site: siteOf, color: (p) => p.color,
    season: (p) => p.season ?? 'Unknown season', seasonColor: (p) => p.seasonColor ?? INK.muted,
    options: (members) => ({ errorsInVariables: true, lambda, bootstrapGroups: monthBlocks(members.map((p) => p.rec ?? {})) }),
  }) : [], [valid, grouping, lambda, fitting]) // eslint-disable-line react-hooks/exhaustive-deps
  const st = useMemo(() => fitting ? regression(valid.map((p) => p.x), valid.map((p) => p.y), { errorsInVariables: true, lambda, bootstrapGroups: monthBlocks(valid.map((p) => p.rec ?? {})) }) : null,
    [valid, lambda, fitting])
  const xs = valid.map((p) => p.x), ys = valid.map((p) => p.y)
  const dom = domain ?? span(sameScale ? [...xs, ...ys] : xs)
  const yDom = sameScale ? dom : yDomain ?? span(ys)
  // the same scales until the ranges change, so the dots and the hit index below survive a re-render (a tooltip)
  const x = useMemo(() => d3.scaleLinear().domain(dom).range([0, iw]), [dom.join(), iw]) // eslint-disable-line react-hooks/exhaustive-deps
  const y = useMemo(() => d3.scaleLinear().domain(yDom).range([ih, 0]), [yDom.join(), ih]) // eslint-disable-line react-hooks/exhaustive-deps
  const [x0, x1] = dom
  const line = (s: number, b: number) => ({ x1: x(x0), y1: y(s * x0 + b), x2: x(x1), y2: y(s * x1 + b) })
  // a filter hovered or pinned on another chart dims this one only when this one draws it too (FocusLayer below:
  // the hover is read there, so it re-renders the one focused dot, not the plot and its fits)
  const keys = useMemo(() => new Set(valid.map((p) => (p.id ? baseFilterId(p.id) : ''))), [valid])
  const dotsRef = useRef<SVGGElement>(null)
  const keyOf = (p: CrossPoint) => (p.id ? baseFilterId(p.id) : null)
  // the dots take no pointer events (HitLayer finds the one under the pointer), and do not follow the hover
  const xy = useMemo(() => valid.map((p) => [x(p.x), y(p.y)] as const), [valid, x, y])
  const dots = useMemo(() => valid.map((p, i) => {
    const key = keyOf(p)
    const dim = !!hl.selected && !!key && !hl.selected.has(key)
    return (
      <circle key={i} cx={xy[i][0]} cy={xy[i][1]} r={3.2} fill={p.hollow ? 'none' : p.color}
        stroke={p.hollow ? p.color : dotRim().stroke} strokeWidth={p.hollow ? 1 : dotRim().strokeWidth}
        {...dotFill(dim ? 'dimmed' : 'standard', p.opacity ?? 1)} strokeOpacity={(dim ? 0.3 : 1) * (p.opacity ?? 1)}
        {...(key ? fidProps(key) : {})} />
    )
  }), [valid, xy, hl.selected])
  const summary = useRegressionSummary({ stats: st, groups: siteFits ? fitGroups : undefined, width: w - 24, enabled: fitting })
  const top = m.top + (summary.height ? summary.height + 14 : 0)
  const height = side + top + m.bottom
  useEffect(() => {
    const node = brushRef.current
    if (!node) return
    const selection = d3.select(node)
    if (!brushing) { selection.selectAll('*').remove(); selection.on('.brush', null); return }
    const brush = d3.brush<unknown>().extent([[0, 0], [iw, ih]]).on('end', (event: d3.D3BrushEvent<unknown>) => {
      if (!event.sourceEvent) return
      if (!event.selection) { hl.setSelected(null); return }
      const [[x0, y0], [x1, y1]] = event.selection as [[number, number], [number, number]]
      const ids = new Set(valid.flatMap(p => p.id && x(p.x) >= x0 && x(p.x) <= x1 && y(p.y) >= y0 && y(p.y) <= y1 ? [baseFilterId(p.id)] : []))
      hl.setSelected(ids.size ? ids : null)
      selection.call(brush.move, null)
    })
    selection.call(brush)
    styleBrush(selection)
    return () => { selection.on('.brush', null); selection.selectAll('*').remove() }
    // Scales are recreated from these domains on each render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [brushing, valid, iw, ih, x.domain().join(), y.domain().join()])
  return (
    <div ref={ref} style={{ position: 'relative' }}>
      {brushable && brush === undefined && <div className="frame-controls"><CrossplotBrushControls brush={[ownBrush, setOwnBrush]} /></div>}
      {/* its points carry a site but no season, so a per-season grouping would only draw the pooled fit */}
      {(multiSite || seasons) && !groupingOverride && <FitGroupingControl only={['pooled', ...(multiSite ? ['site' as const] : []), ...(seasons ? ['season' as const] : [])]}
        value={grouping === 'season' && !seasons ? 'pooled' : grouping} />}
      {/* centred in its panel: a square plot in a wide frame leaves even margins either side */}
      <svg width={w} height={height} fontFamily={FONT.family} style={{ display: 'block', margin: '0 auto' }}>
        {title && <text x={m.left} y={14} fontSize={12.5} fontWeight={600} fill={INK.text}>{title}</text>}
        <g transform="translate(12,22)">{summary.node}</g>
        <g className="regression-plot" transform={`translate(${m.left},${top})`}>
          <defs><clipPath id={clipId}><rect width={iw} height={ih} /></clipPath></defs>
          <YAxis scale={y} x={0} label={yLabel} wrapLabel={brushable} tickCount={5} gridWidth={iw} labelX={-42} format={(v) => String(+(+v).toFixed(2))} pick={pick === true || pick === 'y' ? 'y' : undefined} />
          <XAxis scale={x} y={ih} label={xLabel} wrapLabel={brushable} tickCount={5} format={(v) => String(+(+v).toFixed(2))} labelDy={38} pick={pick === true || pick === 'x' ? 'x' : undefined} />
          <g clipPath={`url(#${clipId})`}>
            {background?.(x, y)}
            {sameScale && <IdentityLine {...line(1, 0)} />}
            <g ref={dotsRef} className="scatter-dots" pointerEvents="none">{dots}</g>
            <FocusLayer keys={keys} items={valid} idOf={(p) => (p.id ? baseFilterId(p.id) : undefined)} groupRef={dotsRef}
              render={(p, pinned) => <>
                <circle cx={x(p.x)} cy={y(p.y)} r={6} fill={p.hollow ? 'none' : p.color} stroke={INK.text} strokeWidth={2}
                  {...dotFill('standard', p.opacity ?? 1)} pointerEvents="none" />
                {pinned && <PinRing cx={x(p.x)} cy={y(p.y)} r={6} />}
              </>} />
            {fitting && (siteFits ? <RegressionGroupLines groups={fitGroups} x={x} y={y} width={iw} height={ih} />
              : <RegressionLines stats={st} x={x} y={y} domain={dom} width={iw} height={ih} />)}
            {extraLines.map((l) => <g key={l.label}><RegressionBand stats={l.stats} method={l.method} color={l.color} x={x} y={y} domain={dom} /><line {...line(l.slope, l.intercept)} stroke={l.color} strokeWidth={1.8} strokeDasharray={l.dash} /></g>)}
            {/* hover, click and right-click find the nearest filter; a click on empty plot lets go of the pin (lib/highlight) */}
            <HitLayer items={valid} xy={xy} width={iw} height={ih}
              onHover={(e, p) => { tip.show(e, p.rec || p.open ? [...p.tip, 'click to pin it on every chart and open its record'] : p.tip); hl.setHover(keyOf(p)) }}
              onLeave={() => { tip.hide(); hl.setHover(null) }}
              onPick={(p) => { const key = keyOf(p); if (p.open) p.open(); else if (p.rec) hl.openRecord(p.rec); if (key) hl.setPinned(key) }}
              onMenu={(e, p) => { const key = keyOf(p); if (key) menu.openMenu(e, { ids: [key], title: key }) }} />
          </g>
          <g ref={brushRef} className="brush-layer" />
        </g>
      </svg>
      {siteFits && <ChartAside><p className="chart-note">Fits use each {noun}'s paired filters and observed x range. {noun[0].toUpperCase() + noun.slice(1)}s with fewer than three pairs or constant values have no fitted line. Point colours retain the selected grouping.</p></ChartAside>}
      {handNote && <ChartAside><p className="chart-note">{handNote}</p></ChartAside>}
      {tip.node}
    </div>
  )
}

/** the crossplot brush on/off, per tile; a chart drawing several crossplots holds one for all of them */
export function useCrossplotBrush(): [boolean, (on: boolean) => void] {
  const [setting, set] = useTileSetting<string>('crossplot.brush', '0')
  return [setting === '1', (on) => set(on ? '1' : '0')]
}

/** the brush toggle and, while filters are selected, what to do with them; in the chart's options row or above the plot */
export function CrossplotBrushControls({ brush: [on, setOn] }: { brush: [boolean, (on: boolean) => void] }) {
  const hl = useHighlight(NO_KEYS)
  const filterEdits = useFilterEdits()
  return <>
    <Toggle label="Brush select" checked={on} onChange={setOn} title="Drag a rectangle to select several filters; turn off to open individual records" />
    {hl.selected && <><span className="control">{hl.selected.size} selected</span>
      <button type="button" className="btn" onClick={() => filterEdits.openGroup({ title: 'Selected crossplot filters', ids: [...hl.selected!], fields: [] })}>Review selected</button>
      <button type="button" className="btn" onClick={() => { filterEdits.keep(hl.selected!); hl.setSelected(null) }}>Keep selected</button>
      <button type="button" className="btn" onClick={() => { filterEdits.drop(hl.selected!); hl.setSelected(null) }}>Exclude selected</button>
      <button type="button" className="btn quiet" onClick={() => hl.setSelected(null)}>Clear selection</button></>}
  </>
}
