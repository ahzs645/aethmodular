import { useId } from 'react'
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

/** the readout's lines (title, n and R², each fit's equation, CIs) and its hover note, shared by the SVG box and the text form */
function useReadout(stats: RegressionStats | null, title?: string, color?: string) {
  const { methods, showCI } = useRegressions()
  if (!stats) return null
  const fits = fitsFor(stats, methods)
  const missingYork = methods.includes('york') && !fits.some(f => f.method === 'york')
  const firstCI = showCI ? fits.map(f => confidence(stats, f.method)).find(Boolean) : null
  const lines = [...(title ? [{ text: title, color: color ?? INK.text }] : []), { text: `n = ${stats.n} · R² = ${fmt(stats.r2, 3)}`, color: INK.text },
    ...(firstCI ? [{ text: firstCI.description.includes('month-block') ? '95% CI · months within site' : '95% CI · independent pairs assumed', color: INK.muted }] : []), ...fits.flatMap(f => {
    const ci = showCI ? confidence(stats, f.method) : null
    return [{ text: equation(f), color: color ?? f.color }, ...(showCI ? ci ? [
      { text: `95% slope [${fmt(ci.slope[0], 3)}, ${fmt(ci.slope[1], 3)}]`, color: color ?? f.color },
      { text: `95% intercept [${fmt(ci.intercept[0], 3)}, ${fmt(ci.intercept[1], 3)}]`, color: color ?? f.color },
    ] : [{ text: '95% CI unavailable (data / resamples)', color: INK.muted }] : [])]
  }),
    ...(missingYork ? [{ text: 'York unavailable: check uncertainties', color: INK.muted }] : [])]
  const note = [fits.find(f => f.method === 'york')?.assumption, ...(showCI ? fits.map(f => confidence(stats, f.method)?.description ?? 'CI unavailable: requires at least 8 pairs, 6 months per site for block resampling, and 90% valid resamples') : [])].filter(Boolean).join('; ')
  return { lines, note }
}

export function RegressionReadout({ stats, x = 10, y = 10, width = 280, title, color }: { stats: RegressionStats | null; x?: number; y?: number; width?: number; title?: string; color?: string }) {
  const r = useReadout(stats, title, color)
  if (!r) return null
  const { lines, note } = r
  // a narrow panel shrinks the monospace text to fit its box instead of letting the SVG edge clip it
  const fontSize = Math.max(8, Math.min(10.5, (width - 16) / (0.61 * Math.max(...lines.map(l => l.text.length)))))
  const step = Math.round(fontSize * 1.43)
  return <g transform={`translate(${x},${y})`} fontFamily={FONT.mono} fontSize={fontSize} pointerEvents="none">
    <rect width={width} height={lines.length * step + 12} rx={5} fill="#fff" fillOpacity={0.94} stroke={INK.border}/>
    {lines.map((l, i) => <text key={i} x={8} y={step + 3 + i * step} fill={l.color}>{l.text}</text>)}
    <title>{note}</title>
  </g>
}

/**
 * The same readout as text under the chart, for a phone: inside the plot the box covers points
 * and runs off a narrow panel. Charts draw this instead of RegressionReadout when
 * useNarrowScreen() holds (export keeps it: it is legend-row text).
 */
export function RegressionReadoutText({ stats, title, color }: { stats: RegressionStats | null; title?: string; color?: string }) {
  const r = useReadout(stats, title, color)
  if (!r) return null
  return <p className="fit-readout legend" data-export="text" title={r.note || undefined}>
    {r.lines.map((l, i) => <span key={i} style={{ color: l.color }}>{l.text}</span>)}
  </p>
}

export function RegressionGroupLines({ groups, ...props }: Omit<Parameters<typeof RegressionLines>[0], 'stats' | 'color' | 'domain'> & { groups: RegressionGroup[] }) {
  return <>{groups.map(group => <g key={group.name} data-fit-site={group.name}>
    <RegressionLines {...props} stats={group.stats} color={group.color} label={group.name} domain={group.domain} />
  </g>)}</>
}

/** Readouts below the axes stay legible and are retained in SVG / PNG exports. */
export function regressionGroupReadoutLayout(count: number, width: number, methods: readonly Estimator[], showCI: boolean) {
  const columns = Math.max(1, Math.min(2, Math.floor(width / 290)))
  const cellWidth = (width - (columns - 1) * 12) / columns
  const rowHeight = 70 + methods.length * (showCI ? 45 : 15)
  return { columns, cellWidth, rowHeight, height: Math.ceil(count / columns) * rowHeight }
}

export function RegressionGroupReadouts({ groups, width, x = 0, y = 0 }: { groups: RegressionGroup[]; width: number; x?: number; y?: number }) {
  const { methods, showCI } = useRegressions()
  const layout = regressionGroupReadoutLayout(groups.length, width, methods, showCI)
  return <g transform={`translate(${x},${y})`}>{groups.map((group, i) => <RegressionReadout key={group.name}
    stats={group.stats} title={group.name} color={group.color} width={layout.cellWidth}
    x={(i % layout.columns) * (layout.cellWidth + 12)} y={Math.floor(i / layout.columns) * layout.rowHeight} />)}</g>
}
