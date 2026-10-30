import { useMemo, type ReactNode } from 'react'
import { ChartFrame } from '@/components/ChartFrame'
import { FacetGrid } from '@/components/FacetGrid'
import { Legend, useLegend } from '@/components/Legend'
import { SweepPlot, type SweepSeries } from '@/components/SweepPlot'
import { useChartOptions } from '@/lib/chartOptions'
import { f2 } from '@/lib/format'
import { useMac } from '@/lib/mac'
import { shortSeason } from '@/lib/seasons'
import { num } from '@/lib/table'
import { rows } from '../../common'
import { GROUPS, baselinesOf, protocolK, sharedDomain, type Data } from './common'

/**
 * The Addis line against the number of PLS components k, one panel per Addis group and one line per
 * baseline: the median over the 100 site-disjoint splits with its IQR, the frozen protocol's k ringed.
 * `cohort` is the standard all-IMPROVE lot-251 model (notebook [3]) or the 500 seasonal analogs, each
 * judged on its own season ([11]). Deming against HIPS Fabs/MAC: the intercept does not move with the MAC,
 * the slope is re-expressed at the reader's (× MAC/10; the workflow's λ is fixed at MAC 10 and scales as MAC²).
 */
export function AddisK({ data, cohort, id, title, subtitle, provenance }: {
  data: Data; cohort: 'standard' | 'analog'; id: string
  title: (s: { show: 'intercept' | 'slope'; mac: string }) => string
  subtitle: string; provenance: ReactNode
}) {
  const m = useMac()
  const lg = useLegend()
  const all = rows(data, 'addis_k').filter((r) => r.cohort === cohort)
  const hasSlope = all.some((r) => num(r.slope) != null)
  const o = useChartOptions(id, [
    { key: 'show', label: 'Show', default: 'intercept', when: () => hasSlope,
      choices: [{ value: 'intercept', label: 'Intercept' }, { value: 'slope', label: `Deming slope (${m.label})` }] },
  ])
  const show = (o.values.show as 'intercept' | 'slope') ?? 'intercept'
  const baselines = baselinesOf(data)

  const panels = useMemo(() => GROUPS.map((g) => {
    const series: SweepSeries[] = baselines.filter((b) => lg.show(b.label)).map((b) => {
      const rs = all.filter((r) => r.baseline === b.key && r.set === g).sort((p, q) => (num(p.k) ?? 0) - (num(q.k) ?? 0))
      // intercepts are MAC-free; a slope against Fabs/10 reads × MAC/10 at the reader's MAC
      const v = (c: string) => (r: (typeof rs)[number]) => (show === 'slope' ? m.slope(num(r[c])) : num(r[c]))
      return {
        id: b.key, label: b.label, color: b.color, group: b.label,
        points: rs.map((r) => ({ x: num(r.k) ?? 0, y: v(show)(r), lo: v(`${show}_q25`)(r), hi: v(`${show}_q75`)(r),
          tip: [`${b.label} · ${shortSeason(g)}${cohort === 'analog' ? ' analogs' : ''}`, `k = ${r.k}`,
            `${show === 'slope' ? `Deming slope (${m.label})` : 'Intercept (µg/m³)'}: ${f2(v(show)(r))}${num(r[`${show}_q25`]) != null ? ` (IQR ${f2(v(`${show}_q25`)(r))} to ${f2(v(`${show}_q75`)(r))})` : ''}`,
            ...(num(r.n_median) != null ? [`${r.n_median} Addis filters`] : [])] })),
      }
    })
    const chosen = baselines.flatMap((b) => { const k = protocolK(data, b.key, cohort, g); return k != null && lg.show(b.label) ? [{ series: b.key, x: k }] : [] })
    return { g, series, chosen }
  }), [all, baselines, cohort, data, lg, m, show])

  const yDomain = sharedDomain(panels.flatMap((p) => p.series.flatMap((s) => s.points.flatMap((q) => [q.y, q.lo, q.hi]))), [show === 'slope' ? 1 : 0])
  const yLabel = show === 'slope' ? `Addis Deming slope (vs Fabs/${m.mac})` : 'Addis Deming intercept (µg/m³)'
  return (
    <ChartFrame id={id} title={title({ show, mac: m.label })} subtitle={subtitle} provenance={provenance} controls={o.controls()}>
      <FacetGrid items={panels} minCell={250} maxCols={4} label={(p) => p.g} render={(p, w) =>
        <SweepPlot series={p.series} chosen={p.chosen} width={w} height={240} title={cohort === 'analog' ? `${shortSeason(p.g)} analogs` : p.g}
          xLabel="PLS components k" yLabel={yLabel} reference={show === 'slope' ? 1 : 0} yDomain={yDomain} dim={lg.dim} />} />
      <Legend {...lg.props} items={baselines.map((b) => ({ label: b.label, color: b.color, shape: 'line' as const }))}
        note={all.some((r) => num(r.intercept_q25) != null) ? 'median and IQR over 100 splits; ringed: the protocol k' : 'median over 100 splits (the notebook displayed seven k); ringed: the protocol k'} />
    </ChartFrame>
  )
}
