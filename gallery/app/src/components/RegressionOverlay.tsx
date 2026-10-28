import { useId, useLayoutEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { useSingleView } from '@/lib/singleView'
import { useExportRender } from '@/lib/exportRender'
import { equation, fitsFor, useRegressions } from '@/lib/regressionPrefs'
import { fmt, type RegressionStats } from '@/lib/stats'
import { FONT, INK } from '@/lib/theme'
import { confidence, lineInterval } from '@/lib/regressionCI'
import type { Estimator } from '@/lib/regressionMethods'
import type { RegressionGroup } from '@/lib/siteRegression'

export function RegressionBand({ stats, method, color, x, y, domain }: { stats: RegressionStats | null; method: Estimator; color: string; x: (v: number) => number; y: (v: number) => number; domain: number[] }) {
  const { showCI } = useRegressions()
  if (!showCI) return null
  const ci = confidence(stats, method)
  if (!ci) return null
  const points = Array.from({ length: 61 }, (_, i) => {
    const xv = domain[0] + (domain[1] - domain[0]) * i / 60
    const [lo, hi] = lineInterval(ci, xv)
    return { x: x(xv), lo: y(lo), hi: y(hi) }
  }).filter(p => [p.x, p.lo, p.hi].every(Number.isFinite))
  if (points.length < 2) return null
  const d = `M${points.map(p => `${p.x},${p.lo}`).join('L')}L${points.slice().reverse().map(p => `${p.x},${p.hi}`).join('L')}Z`
  return <path data-fit-uncertainty={method} d={d} fill={color} fillOpacity={0.14} pointerEvents="none"><title>{ci.description}; pointwise line confidence band</title></path>
}

export function RegressionLines({ stats, x, y, domain, width, height, color, label }: { stats: RegressionStats | null; x: (v: number) => number; y: (v: number) => number; domain: number[]; width: number; height: number; color?: string; label?: string }) {
  const { methods } = useRegressions()
  const id = useId()
  const fits = fitsFor(stats, methods)
  return <g pointerEvents="none"><defs><clipPath id={id}><rect width={width} height={height} /></clipPath></defs><g clipPath={`url(#${id})`}>
    {fits.map(fit => <RegressionBand key={`band-${fit.method}`} stats={stats} method={fit.method} color={color ?? fit.color} x={x} y={y} domain={domain} />)}
    {fits.map(fit => {
      // Sample in data coordinates so the central fit also follows log axes.
      const points = Array.from({ length: 61 }, (_, i) => {
        const xv = domain[0] + (domain[1] - domain[0]) * i / 60
        return [x(xv), y(fit.slope * xv + fit.intercept)]
      }).filter(p => p.every(Number.isFinite))
      return points.length > 1 ? <path key={fit.method} data-regression={fit.method} d={`M${points.map(p => p.join(',')).join('L')}`} fill="none" stroke={color ?? fit.color} strokeWidth={2} strokeDasharray={fit.dash}><title>{label ? `${label} · ` : ''}{equation(fit)}{fit.assumption ? ` · ${fit.assumption}` : ''}</title></path> : null
    })}
  </g></g>
}

/** Shared content for the summary and specialised SVG legends. */
function readout(stats: RegressionStats | null, methods: Estimator[], showCI: boolean, title?: string, color?: string) {
  if (!stats) return null
  const fits = fitsFor(stats, methods)
  const missingYork = methods.includes('york') && !fits.some(f => f.method === 'york')
  const firstCI = showCI ? fits.map(f => confidence(stats, f.method)).find(Boolean) : null
  const lines: { text: string; color: string; method?: Estimator }[] = [...(title ? [{ text: title, color: color ?? INK.text }] : []), { text: `n = ${stats.n} · R² = ${fmt(stats.r2, 3)}`, color: INK.text },
    ...(firstCI ? [{ text: firstCI.description.includes('month-block') ? '95% CI · months within site' : '95% CI · independent pairs assumed', color: INK.muted }] : []), ...fits.flatMap(f => {
    const ci = showCI ? confidence(stats, f.method) : null
    return [{ text: equation(f), color: color ?? f.color, method: f.method }, ...(showCI ? ci ? [
      { text: `95% slope [${fmt(ci.slope[0], 3)}, ${fmt(ci.slope[1], 3)}]`, color: color ?? f.color, method: f.method },
      { text: `95% intercept [${fmt(ci.intercept[0], 3)}, ${fmt(ci.intercept[1], 3)}]`, color: color ?? f.color, method: f.method },
    ] : [{ text: '95% CI unavailable (data / resamples)', color: INK.muted }] : [])]
  }),
    ...(missingYork ? [{ text: 'York unavailable: check uncertainties', color: INK.muted }] : [])]
  const note = [fits.find(f => f.method === 'york')?.assumption, ...(showCI ? fits.map(f => confidence(stats, f.method)?.description ?? 'CI unavailable: requires at least 8 pairs, 6 months per site for block resampling, and 90% valid resamples') : [])].filter(Boolean).join('; ')
  return { lines, note }
}

export function RegressionReadout({ stats, x = 10, y = 10, width = 280, title, color }: { stats: RegressionStats | null; x?: number; y?: number; width?: number; title?: string; color?: string }) {
  const { methods, showCI } = useRegressions()
  const r = readout(stats, methods, showCI, title, color)
  if (!r) return null
  const { lines, note } = r
  // a narrow panel shrinks the monospace text to fit its box instead of letting the SVG edge clip it
  const fontSize = Math.max(8, Math.min(10.5, (width - 16) / (0.61 * Math.max(...lines.map(l => l.text.length)))))
  const step = Math.round(fontSize * 1.43)
  return <g transform={`translate(${x},${y})`} fontFamily={FONT.mono} fontSize={fontSize} pointerEvents="none">
    <rect width={width} height={lines.length * step + 12} rx={5} fill={INK.panel} fillOpacity={0.94} stroke={INK.border}/>
    {lines.map((l, i) => <text key={i} x={8} y={step + 3 + i * step} fill={l.color} data-regression-color={l.method}>{l.text}</text>)}
    <title>{note}</title>
  </g>
}

const SUMMARY_FONT = 11
const SUMMARY_STEP = 16
const SUMMARY_GAP = 12

/** Wrap SVG text at a readable size, including long group names and CI notes. */
function wrapSummary(text: string, width: number) {
  const limit = Math.max(1, Math.floor((width - 16) / (SUMMARY_FONT * 0.62)))
  const lines: string[] = []
  let line = ''
  for (const word of text.split(/\s+/)) {
    if (line && line.length + word.length + 1 > limit) { lines.push(line); line = '' }
    // IDs and other unbroken labels must also fit the box.
    let rest = word
    while (rest.length > limit) { lines.push(rest.slice(0, limit)); rest = rest.slice(limit) }
    line = line ? `${line} ${rest}` : rest
  }
  if (line) lines.push(line)
  return lines
}

/**
 * A single placement rule for paired-data charts: reserve a summary above the
 * axes at every width. n / R² precede one equation per line. Multiple fits use
 * the same boxes, stacked on phones and up to two columns on wide panels.
 * The returned height is part of the SVG, so exports keep the exact layout.
 */
export function useRegressionSummary({ stats, groups, width, enabled = true, panel }: {
  stats: RegressionStats | null; groups?: RegressionGroup[]; width: number; enabled?: boolean
  /** a faceted chart's panel title: in the sidebar card it names whose fit this is, and pointing at it outlines that panel */
  panel?: { title: string; color?: string } | null
}) {
  const { methods, showCI } = useRegressions()
  // Where the summary goes, found from the DOM by a probe left in the chart:
  //   chart  above the axes (everywhere, and in an export as on screen)
  //   side   the chart in single-chart view (lib/singleView): the sidebar card, at its width; the plot keeps the height
  //   data   an export setting it inside the plot (lib/exportRender): hidden here, read by lib/export
  const single = useSingleView()
  const render = useExportRender()
  const probe = useRef<SVGGElement>(null)
  const [place, setPlace] = useState<'chart' | 'side' | 'data'>('chart')
  useLayoutEffect(() => {
    const el = probe.current
    const now = el && render?.panel.contains(el) ? (render.statsInside ? 'data' : 'chart')
      : el && single.active && single.slot && el.closest('[data-single-active]') ? 'side' : 'chart'
    if (now !== place) setPlace(now)
  })
  const slot = place === 'side' ? single.slot : null
  if (slot) width = Math.max(200, slot.clientWidth || 320)
  const items = !enabled || !methods.length ? [] : groups ?? [{ stats, name: undefined, color: undefined }]
  const columns = Math.max(1, Math.min(2, items.length, Math.floor(width / 290)))
  const cellWidth = (width - (columns - 1) * SUMMARY_GAP) / columns
  const boxes = items.flatMap(group => {
    const r = readout(group.stats, methods, showCI, group.name, group.color)
    if (!r) return []
    const lines = r.lines.flatMap(line => wrapSummary(line.text, cellWidth).map(text => ({ ...line, text })))
    // Reserve the same minimum height in adjacent pooled panels even when a
    // selected method or its CI is unavailable in one panel. CI adds one
    // shared resampling description, plus two interval lines per method.
    const minLines = (group.name ? 1 : 0) + 1 + methods.length * (showCI ? 3 : 1) + (showCI ? 1 : 0)
    return [{ ...r, raw: r.lines, lines, name: group.name, color: group.color, height: Math.max(minLines, lines.length) * SUMMARY_STEP + 12 }]
  })
  let height = 0
  let rowY = 0
  const positions = boxes.map((_, i) => {
    if (i % columns === 0) {
      rowY = height + (i ? SUMMARY_GAP : 0)
      const rowHeight = Math.max(...boxes.slice(i, i + columns).map(b => b.height))
      height = rowY + rowHeight
    }
    return { x: (i % columns) * (cellWidth + SUMMARY_GAP), y: rowY }
  })
  const node = boxes.length ? <g className="regression-summary" role="group" aria-label="Regression statistics" fontFamily={FONT.mono} fontSize={SUMMARY_FONT} pointerEvents="none">
    {boxes.map((box, i) => <g key={box.name ?? 'pooled'} transform={`translate(${positions[i].x},${positions[i].y})`} data-fit-site={box.name} data-export-color={box.color}>
      <rect width={cellWidth} height={box.height} rx={5} fill={INK.panel} stroke={INK.border} />
      {box.lines.map((line, j) => <text key={j} x={8} y={SUMMARY_STEP + 3 + j * SUMMARY_STEP} fill={line.color} data-regression-color={line.method}>{line.text}</text>)}
      {box.note && <title>{box.note}</title>}
    </g>)}
  </g> : null
  if (!node) return { height, node }
  if (place === 'data') return { height: 0, node: <><g ref={probe} /><g data-export-stats="" display="none">
    {boxes.map((box) => <g key={box.name ?? 'pooled'} data-fit-site={box.name} data-export-color={box.color}>
      {box.raw.map((line, j) => <text key={j} fill={line.color} data-regression-color={line.method} data-line={line.text === box.name ? 'name' : undefined}>{line.text}</text>)}
    </g>)}
  </g></> }
  if (slot) {
    const outline = (on: boolean) => probe.current?.closest('svg')?.parentElement?.classList.toggle('fit-pointed', on)
    const side = <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>{node}</svg>
    return { height: 0, node: <><g ref={probe} />{createPortal(panel
      ? <div className="regression-summary-side" onMouseEnter={() => outline(true)} onMouseLeave={() => outline(false)}>
          <p className="side-fit-title">{panel.color && <span className="swatch" style={{ background: panel.color }} />}{panel.title}</p>
          {side}
        </div>
      : <div className="regression-summary-side">{side}</div>, slot)}</> }
  }
  return { height, node: <><g ref={probe} />{node}</> }
}

export function RegressionGroupLines({ groups, ...props }: Omit<Parameters<typeof RegressionLines>[0], 'stats' | 'color' | 'domain'> & { groups: RegressionGroup[] }) {
  return <>{groups.map(group => <g key={group.name} data-fit-site={group.name} data-export-color={group.color}>
    <RegressionLines {...props} stats={group.stats} color={group.color} label={group.name} domain={group.domain} />
  </g>)}</>
}
