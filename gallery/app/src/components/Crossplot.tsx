import { useId, useMemo, useRef, type ReactNode } from 'react'
import { IdentityLine } from '@/components/RefLines'
import * as d3 from 'd3'
import { RegressionLines, RegressionReadout, RegressionBand, RegressionGroupLines, RegressionGroupReadouts, regressionGroupReadoutLayout } from '@/components/RegressionOverlay'
import { XAxis, YAxis } from '@/components/Axes'
import { useDimensions } from '@/hooks/useGalleryData'
import { useTooltip } from '@/hooks/useTooltip'
import { groupedRegressions, type FitGrouping } from '@/lib/siteRegression'
import { monthBlocks } from '@/lib/regressionCI'
import { regression, type RegressionStats } from '@/lib/stats'
import { useRegressions, FitGroupingControl } from '@/lib/regressionPrefs'
import type { Estimator } from '@/lib/regressionMethods'
import { baseFilterId, useHighlight, type FilterRecord } from '@/lib/highlight'
import { FONT, INK } from '@/lib/theme'

/** One filter on a crossplot. `rec` opens the record drawer; `id` links hover and pin across charts. */
export interface CrossPoint {
  x: number; y: number; color: string; hollow?: boolean; tip: string[]; id?: string; rec?: FilterRecord
  /** opacity multiplier, e.g. a legend dimming every series but the hovered one */
  opacity?: number
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
export function Crossplot({ points, title, xLabel, yLabel, maxSide = 360, domain, fit = true, errorRatio, extraLines = [], grouping: groupingOverride, background }: {
  points: CrossPoint[]
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
  /** drawn under the points on the same scales, e.g. a density of the wider population */
  background?: (x: d3.ScaleLinear<number, number>, y: d3.ScaleLinear<number, number>) => ReactNode
}) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const tip = useTooltip(ref)
  const hl = useHighlight()
  const m = { top: 24, right: 14, bottom: 46, left: 56 }
  const side = Math.max(160, Math.min(width - m.left - m.right, maxSide))
  const iw = side, ih = side, w = side + m.left + m.right, height = side + m.top + m.bottom
  const clipId = useId()
  const valid = points.filter((p) => p.x != null && p.y != null && Number.isFinite(p.x) && Number.isFinite(p.y))
  const lambda = errorRatio ?? 1
  const prefs = useRegressions()
  const { methods, showCI } = prefs
  const fitting = fit && groupingOverride !== 'none'
  const grouping: FitGrouping = groupingOverride && groupingOverride !== 'none' ? groupingOverride : prefs.grouping
  const multiSite = fitting && new Set(valid.map((p) => p.rec?.site)).size > 1
  const siteFits = multiSite && grouping === 'site' && methods.length > 0   // its points carry no season
  const fitGroups = useMemo(() => fitting ? groupedRegressions(valid, grouping, {
    x: (p) => p.x, y: (p) => p.y, site: (p) => p.rec?.site ?? 'Unknown site', color: (p) => p.color,
    options: (members) => ({ errorsInVariables: true, lambda, bootstrapGroups: monthBlocks(members.map((p) => p.rec ?? {})) }),
  }) : [], [valid, grouping, lambda, fitting])
  const st = fitting ? regression(valid.map((p) => p.x), valid.map((p) => p.y), { errorsInVariables: true, lambda, bootstrapGroups: monthBlocks(valid.map((p) => p.rec ?? {})) }) : null
  const xs = valid.map((p) => p.x), ys = valid.map((p) => p.y)
  const dom = domain ?? d3.scaleLinear().domain([Math.min(0, d3.min([...xs, ...ys]) ?? 0), (d3.max([...xs, ...ys]) ?? 1) * 1.05]).nice().domain() as [number, number]
  const x = d3.scaleLinear().domain(dom).range([0, iw])
  const y = d3.scaleLinear().domain(dom).range([ih, 0])
  const [x0, x1] = dom
  const line = (s: number, b: number) => ({ x1: x(x0), y1: y(s * x0 + b), x2: x(x1), y2: y(s * x1 + b) })
  return (
    <div ref={ref} style={{ position: 'relative' }}>
      {multiSite && !groupingOverride && <FitGroupingControl />}
      <svg width={w} height={height} fontFamily={FONT.family}>
        <text x={m.left} y={14} fontSize={12.5} fontWeight={600} fill={INK.text}>{title}</text>
        <g transform={`translate(${m.left},${m.top})`}>
          <defs><clipPath id={clipId}><rect width={iw} height={ih} /></clipPath></defs>
          <YAxis scale={y} x={0} label={yLabel} tickCount={5} gridWidth={iw} labelX={-42} format={(v) => String(+(+v).toFixed(2))} />
          <XAxis scale={x} y={ih} label={xLabel} tickCount={5} format={(v) => String(+(+v).toFixed(2))} labelDy={38} />
          <g clipPath={`url(#${clipId})`}>
            <rect width={iw} height={ih} fill="transparent" onClick={() => hl.setPinned(null)} />
            {background?.(x, y)}
            <IdentityLine {...line(1, 0)} />
            {valid.map((p, i) => {
              const key = p.id ? baseFilterId(p.id) : null
              const focus = !!key && (key === hl.pinnedId || key === hl.hoverId)
              const dim = !!(hl.pinnedId || hl.hoverId) && !focus
              return (
                <circle key={i} cx={x(p.x)} cy={y(p.y)} r={focus ? 6 : 3.2} fill={p.hollow ? 'none' : p.color}
                  stroke={focus ? INK.text : p.color} strokeWidth={focus ? 2 : p.hollow ? 1 : 0.4}
                  fillOpacity={(dim ? 0.25 : 0.7) * (p.opacity ?? 1)} strokeOpacity={(dim ? 0.3 : 1) * (p.opacity ?? 1)} style={{ cursor: p.rec ? 'pointer' : 'default' }}
                  onMouseEnter={(e) => { tip.show(e, p.rec ? [...p.tip, 'click to pin it on every chart and open its record'] : p.tip); if (key) hl.setHover(key) }}
                  onMouseLeave={() => { tip.hide(); hl.setHover(null) }}
                  onClick={() => { if (p.rec) hl.openRecord(p.rec); if (key) hl.setPinned(key) }} />
              )
            })}
            {fitting && (siteFits ? <RegressionGroupLines groups={fitGroups} x={x} y={y} width={iw} height={ih} />
              : <RegressionLines stats={st} x={x} y={y} domain={dom} width={iw} height={ih} />)}
            {extraLines.map((l) => <g key={l.label}><RegressionBand stats={l.stats} method={l.method} color={l.color} x={x} y={y} domain={dom} /><line {...line(l.slope, l.intercept)} stroke={l.color} strokeWidth={1.8} strokeDasharray={l.dash} /></g>)}
          </g>
          {fitting && !siteFits && <RegressionReadout stats={st} x={8} y={6} width={Math.min(280, iw - 12)} />}
        </g>
      </svg>
      {siteFits && <details><summary>Per-site fit statistics ({fitGroups.length} sites)</summary>
        <svg width={w} height={regressionGroupReadoutLayout(fitGroups.length, w, methods, showCI).height}>
          <RegressionGroupReadouts groups={fitGroups} width={w} />
        </svg>
        <p className="chart-note">Fits use each site's paired filters and observed x range. Sites with fewer than three pairs or constant values have no fitted line. Point colours retain the selected grouping.</p>
      </details>}
      {tip.node}
    </div>
  )
}
