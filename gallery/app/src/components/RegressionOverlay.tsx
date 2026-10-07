import { useId } from 'react'
import { equation, fitsFor, useRegressions } from '@/lib/regressionPrefs'
import { fmt, type RegressionStats } from '@/lib/stats'
import { FONT, INK } from '@/lib/theme'
import { confidence, lineInterval } from '@/lib/regressionCI'
import type { Estimator } from '@/lib/regressionMethods'

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

export function RegressionLines({ stats, x, y, domain, width, height }: { stats: RegressionStats | null; x: (v: number) => number; y: (v: number) => number; domain: number[]; width: number; height: number }) {
  const { methods } = useRegressions()
  const id = useId()
  const fits = fitsFor(stats, methods)
  return <g pointerEvents="none"><defs><clipPath id={id}><rect width={width} height={height} /></clipPath></defs><g clipPath={`url(#${id})`}>
    {fits.map(fit => <RegressionBand key={`band-${fit.method}`} stats={stats} method={fit.method} color={fit.color} x={x} y={y} domain={domain} />)}
    {fits.map(fit => {
      // Sample in data coordinates so the central fit also follows log axes.
      const points = Array.from({ length: 61 }, (_, i) => {
        const xv = domain[0] + (domain[1] - domain[0]) * i / 60
        return [x(xv), y(fit.slope * xv + fit.intercept)]
      }).filter(p => p.every(Number.isFinite))
      return points.length > 1 ? <path key={fit.method} data-regression={fit.method} d={`M${points.map(p => p.join(',')).join('L')}`} fill="none" stroke={fit.color} strokeWidth={2} strokeDasharray={fit.dash}><title>{equation(fit)}{fit.assumption ? ` · ${fit.assumption}` : ''}</title></path> : null
    })}
  </g></g>
}

export function RegressionReadout({ stats, x = 10, y = 10, width = 280 }: { stats: RegressionStats | null; x?: number; y?: number; width?: number }) {
  const { methods, showCI } = useRegressions()
  if (!stats) return null
  const fits = fitsFor(stats, methods)
  const missingYork = methods.includes('york') && !fits.some(f => f.method === 'york')
  const firstCI = showCI ? fits.map(f => confidence(stats, f.method)).find(Boolean) : null
  const lines = [{ text: `n = ${stats.n} · R² = ${fmt(stats.r2, 3)}`, color: INK.text },
    ...(firstCI ? [{ text: firstCI.description.includes('month-block') ? '95% CI · months within site' : '95% CI · independent pairs assumed', color: INK.muted }] : []), ...fits.flatMap(f => {
    const ci = showCI ? confidence(stats, f.method) : null
    return [{ text: equation(f), color: f.color }, ...(showCI ? ci ? [
      { text: `95% slope [${fmt(ci.slope[0], 3)}, ${fmt(ci.slope[1], 3)}]`, color: f.color },
      { text: `95% intercept [${fmt(ci.intercept[0], 3)}, ${fmt(ci.intercept[1], 3)}]`, color: f.color },
    ] : [{ text: '95% CI unavailable (data / resamples)', color: INK.muted }] : [])]
  }),
    ...(missingYork ? [{ text: 'York unavailable: check uncertainties', color: INK.muted }] : [])]
  return <g transform={`translate(${x},${y})`} fontFamily={FONT.mono} fontSize={10.5} pointerEvents="none">
    <rect width={width} height={lines.length * 15 + 12} rx={5} fill="#fff" fillOpacity={0.94} stroke={INK.border}/>
    {lines.map((l, i) => <text key={i} x={8} y={18 + i * 15} fill={l.color}>{l.text}</text>)}
    <title>{[fits.find(f => f.method === 'york')?.assumption, ...(showCI ? fits.map(f => confidence(stats, f.method)?.description ?? 'CI unavailable: requires at least 8 pairs, 6 months per site for block resampling, and 90% valid resamples') : [])].filter(Boolean).join('; ')}</title>
  </g>
}
