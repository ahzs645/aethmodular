import { useId, useMemo, useRef, useState, useSyncExternalStore, type ReactNode } from 'react'
import * as d3 from 'd3'
import { XAxis, YAxis } from '@/components/Axes'
import type { AxisRole } from '@/components/AxisFieldPopover'
import { Marker } from '@/components/RefLines'
import { SeasonBands, timeSeasonBands } from '@/components/SeasonBands'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip, type TipBody } from '@/hooks/useTooltip'
import { useFitHeight } from '@/lib/chartFit'
import { stackGeometry, type StackSize } from '@/lib/stacked'
import type { SeasonMeta } from '@/lib/types'
import { FONT, INK } from '@/lib/theme'

/**
 * Panels stacked one above another on ONE x axis, the way the timelines and QC notebooks draw a record: the
 * AERONET and weather timelines (components/DailyLanes), the aethalometer daily panels and agreement over time.
 *
 *   <StackedPanels x={{ type: 'time', domain }} seasons={seasons} readout={(px, x) => …}
 *     panels={[{ key: 'aod', y: { domain: [0, 2], label: 'AOD 500' }, children: ({ x, y }) => <path … /> }, …]} />
 *
 * - one x scale (time, linear or point) for every panel, its axis drawn once at the bottom;
 * - each panel its own y (a domain: StackedPanels builds the scale for the panel's height and draws the axis
 *   with its grid), an optional `title` drawn in the left margin instead of the axis label, and a `readout`
 *   at its top right;
 * - heights: `panelHeight` × each panel's `share` (or its own `height`); in a Dashboard tile the fitted height
 *   (lib/chartFit useFitHeight) is divided by share (lib/stacked stackGeometry). Width is measured
 *   (hooks/useGalleryData useDimensions), so a tile's legend column is kept out of it;
 * - one dashed cursor across every panel and one tooltip: `readout(px, x)` returns where the cursor snaps
 *   (usually the nearest record, lib/stacked nearestIndex) and what the tooltip says. The cursor lives in its
 *   own small store, so moving it re-renders the line, never the panels;
 * - season shading behind every panel (components/SeasonBands, time x only), vertical `xMarkers` through every
 *   panel (labelled in the top one);
 * - `background` and `children` render props per panel get the scales and size; filter points inside go
 *   through components/ScatterLayer with the `tip` they are given, and set `pointer: false` so the panel's
 *   own pointer layer answers instead of the cursor. Lines that must break at gaps: lib/stacked lineRuns.
 *
 * Everything is one <svg>, so lib/export composeFigure captures the whole stack as one figure.
 */

export type StackX =
  | { type: 'time'; domain: readonly [Date, Date] }
  | { type: 'linear'; domain: readonly [number, number]; nice?: boolean }
  | { type: 'point'; domain: readonly string[]; padding?: number }
export type StackScaleOf<X extends StackX> = X extends { type: 'time' } ? d3.ScaleTime<number, number>
  : X extends { type: 'linear' } ? d3.ScaleLinear<number, number> : d3.ScalePoint<string>

export interface StackY {
  domain: readonly [number, number]
  /** round the domain out (d3 nice) */
  nice?: boolean
  label?: string
  /** where the rotated label stands, relative to the axis (YAxis labelX) */
  labelX?: number
  wrapLabel?: boolean
  /** the label opens the shared field picker (YAxis pick) */
  pick?: AxisRole
  /** default 3 */
  tickCount?: number
  format?: (v: any) => string
  /** wraps the drawn axis, e.g. in an AxisChoices offering this panel's own y options */
  around?: (axis: ReactNode) => ReactNode
}

export interface PanelScope<S> {
  key: string
  index: number
  x: S
  y: d3.ScaleLinear<number, number>
  /** the plot's size, px */
  width: number
  height: number
  /** `url(#…)` of a clip path the size of the plot */
  clip: string
  /** the stack's tooltip, for a ScatterLayer or a mark's own hover */
  tip: ReturnType<typeof useTooltip>
}

export interface StackPanel<S> extends StackSize {
  key: string
  y: StackY
  /** a title of the chart's own in the left margin (the weather timeline's swappable lane names) */
  title?: (s: { height: number; left: number }) => ReactNode
  /** a short value at the panel's top right */
  readout?: ReactNode
  /** a line of text just above the panel, from its left edge (leave room: margin.top for the first, `gap` for the others) */
  heading?: ReactNode
  /** under the axis grid and the data (a shaded period) */
  background?: (s: PanelScope<S>) => ReactNode
  /** the panel's marks */
  children?: (s: PanelScope<S>) => ReactNode
  /** the panel moves the shared cursor (default true); false where a ScatterLayer answers the pointer */
  pointer?: boolean
}

export interface StackMarker { at: number | Date; label?: string; color?: string; labelSide?: 'left' | 'right' }

type Margin = { top?: number; right?: number; bottom?: number; left?: number | ((heights: readonly number[]) => number) }

export function StackedPanels<X extends StackX>({ x: xSpec, panels, margin = {}, gap = 14, panelHeight = 92, minPanelHeight = 40,
  xAxis = {}, seasons, seasonHover, xMarkers = [], readout, onLeave, overlay, ariaLabel, className }: {
  x: X
  /** the scales come from `x` (NoInfer: the panels' render props do not widen it) */
  panels: readonly StackPanel<StackScaleOf<NoInfer<X>>>[]
  /** px; `left` may depend on the panels' heights (titles wrapped to fit beside them) */
  margin?: Margin
  gap?: number
  /** a share-1 panel's natural height, px */
  panelHeight?: number
  minPanelHeight?: number
  /** the shared x axis at the bottom; false: none */
  xAxis?: { tickCount?: number; format?: (v: any) => string; label?: string; pick?: AxisRole } | false
  /** season shading behind every panel (a time x only) */
  seasons?: readonly Pick<SeasonMeta, 'name' | 'months' | 'color' | 'from' | 'spans'>[]
  seasonHover?: string | null
  /** vertical lines through every panel, labelled in the top one */
  xMarkers?: readonly StackMarker[]
  /** where the cursor stands for a pointer at `px`, and the tooltip; null: none here */
  readout?: (px: number, x: StackScaleOf<NoInfer<X>>) => { at: number; tip: TipBody } | null
  onLeave?: () => void
  /** marks over the whole stack, in plot coordinates (x from 0, y from the svg's top) */
  overlay?: (s: { x: StackScaleOf<NoInfer<X>>; width: number; tops: readonly number[]; heights: readonly number[] }) => ReactNode
  ariaLabel?: string
  /** on the svg, e.g. 'animated' to ease marks between settings */
  className?: string
}) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const uid = useId().replace(/:/g, '')
  const [cursor] = useState(createCursor)
  const fit = useFitHeight()
  const top = margin.top ?? 8, bottom = margin.bottom ?? (xAxis && xAxis.label ? 46 : 30), right = margin.right ?? 20
  const geo = stackGeometry(panels, { fit, base: panelHeight, gap, top, bottom, min: minPanelHeight })
  const left = typeof margin.left === 'function' ? margin.left(geo.heights) : margin.left ?? 68
  const innerW = Math.max(240, width - left - right)

  // the same scales until the data or the size change, so memoised marks inside (ScatterLayer dots) survive
  const xSig = `${xSpec.type}|${xSpec.domain.map((v) => +v || String(v)).join('|')}|${'nice' in xSpec ? xSpec.nice : ''}|${'padding' in xSpec ? xSpec.padding : ''}`
  const x = useMemo(() => buildX(xSpec, innerW), [xSig, innerW]) as StackScaleOf<X> // eslint-disable-line react-hooks/exhaustive-deps
  const ySig = panels.map((p) => `${p.y.domain[0]},${p.y.domain[1]},${p.y.nice ? 1 : 0}`).join('|'), hSig = geo.heights.join('|')
  const ys = useMemo(() => panels.map((p, i) => {
    const [a, b] = p.y.domain
    const s = d3.scaleLinear().domain(a === b ? [a, a + 1] : [a, b]).range([geo.heights[i], 0])
    return p.y.nice ? s.nice() : s
  }), [ySig, hSig]) // eslint-disable-line react-hooks/exhaustive-deps
  const seasonSig = seasons?.map((s) => `${s.name}:${s.color}:${s.months.join(',')}:${s.from ?? ''}:${s.spans?.length ?? ''}`).join('|') ?? ''
  const bands = useMemo(() => (seasons?.length && xSpec.type === 'time' ? timeSeasonBands(x as d3.ScaleTime<number, number>, seasons, innerW) : []),
    [x, seasonSig, innerW]) // eslint-disable-line react-hooks/exhaustive-deps
  const px = (v: number | Date) => (x as d3.ScaleLinear<number, number>)(v as number)
  const inView = (v: number | Date) => { const p = px(v); return Number.isFinite(p) && p >= 0 && p <= innerW }

  const move = (e: React.MouseEvent<SVGRectElement>) => {
    const r = readout?.(d3.pointer(e)[0], x)
    if (!r) { cursor.set(null); tip.hide(); return }
    cursor.set(r.at)
    tip.show(e, r.tip)
  }

  return (
    <div ref={ref} style={{ position: 'relative' }}>
      <svg width={innerW + left + right} height={geo.total} fontFamily={FONT.family} className={className} role={ariaLabel ? 'img' : undefined} aria-label={ariaLabel}
        onMouseLeave={() => { cursor.set(null); tip.hide(); onLeave?.() }}>
        <defs>{panels.map((p, i) => <clipPath key={p.key} id={`${uid}-${i}`}><rect width={innerW} height={geo.heights[i]} /></clipPath>)}</defs>
        <g transform={`translate(${left},0)`}>
          {panels.map((p, i) => {
            const h = geo.heights[i]
            const s: PanelScope<StackScaleOf<X>> = { key: p.key, index: i, x, y: ys[i], width: innerW, height: h, clip: `url(#${uid}-${i})`, tip }
            return (
              <g key={p.key} transform={`translate(0,${geo.tops[i]})`} data-stack-panel={p.key}>
                {bands.length > 0 && <SeasonBands bands={bands} height={h} hover={seasonHover} />}
                {p.background?.(s)}
                {wrapAxis(p.y, <YAxis scale={ys[i]} x={0} gridWidth={innerW} tickCount={p.y.tickCount ?? 3} label={p.y.label} labelX={p.y.labelX}
                  wrapLabel={p.y.wrapLabel} pick={p.y.pick} format={p.y.format} />)}
                {p.heading != null && <text x={0} y={-10} fontSize={12.5} fontWeight={600} fill={INK.text}>{p.heading}</text>}
                {p.title?.({ height: h, left })}
                {xMarkers.filter((m) => inView(m.at)).map((m, k) => <Marker key={k} orient="v" at={px(m.at)} from={0} to={h}
                  label={i === 0 ? m.label : undefined} color={m.color} labelSide={m.labelSide} />)}
                {p.children?.(s)}
                {p.readout != null && <text x={innerW - 4} y={11} textAnchor="end" fontSize={10.5} fill={INK.muted} pointerEvents="none">{p.readout}</text>}
                {readout && p.pointer !== false && <rect width={innerW} height={h} fill="transparent" onMouseMove={move} />}
              </g>
            )
          })}
          <CursorLine store={cursor} y1={top} y2={geo.total - bottom} />
          {overlay?.({ x, width: innerW, tops: geo.tops, heights: geo.heights })}
          {xAxis && <XAxis scale={x as d3.ScaleLinear<number, number>} y={geo.total - bottom} tickCount={xAxis.tickCount ?? 8} pick={xAxis.pick}
            format={xAxis.format ?? (xSpec.type === 'time' ? MONTH_YEAR : undefined)} label={xAxis.label} />}
        </g>
      </svg>
      {tip.node}
    </div>
  )
}

const wrapAxis = (y: StackY, axis: ReactNode) => (y.around ? y.around(axis) : axis)

const MONTH_YEAR = d3.utcFormat('%b %Y') as (v: any) => string

function buildX(spec: StackX, w: number): StackXScaleAny {
  if (spec.type === 'time') return d3.scaleTime().domain(spec.domain as [Date, Date]).range([0, w])
  if (spec.type === 'point') return d3.scalePoint<string>().domain(spec.domain as string[]).range([0, w]).padding(spec.padding ?? 0.4)
  const s = d3.scaleLinear().domain(spec.domain as [number, number]).range([0, w])
  return spec.nice ? s.nice() : s
}
type StackXScaleAny = d3.ScaleTime<number, number> | d3.ScaleLinear<number, number> | d3.ScalePoint<string>

/** the cursor's x, read by the line alone */
function createCursor() {
  let at: number | null = null
  const ls = new Set<() => void>()
  return {
    get: () => at,
    set: (v: number | null) => { if (v === at) return; at = v; for (const l of ls) l() },
    subscribe: (l: () => void) => { ls.add(l); return () => { ls.delete(l) } },
  }
}

function CursorLine({ store, y1, y2 }: { store: ReturnType<typeof createCursor>; y1: number; y2: number }) {
  const at = useSyncExternalStore(store.subscribe, store.get, store.get)
  return at === null ? null : <line x1={at} x2={at} y1={y1} y2={y2} stroke={INK.text} strokeOpacity={0.35} strokeDasharray="3 3" pointerEvents="none" data-export="skip" />
}
