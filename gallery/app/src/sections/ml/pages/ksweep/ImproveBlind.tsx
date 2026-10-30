import * as d3 from 'd3'
import type { ReactNode } from 'react'
import { ChartFrame } from '@/components/ChartFrame'
import { Legend, useLegend } from '@/components/Legend'
import { SweepPlot, type SweepPoint, type SweepSeries } from '@/components/SweepPlot'
import { f2 } from '@/lib/format'
import { num } from '@/lib/table'
import { rows } from '../../common'
import { Grid } from '../../../research/common/ui'
import { at, baselinesOf, protocolK, type Data } from './common'

/**
 * Notebook [5]: what the standard IMPROVE validation sees over the same k range. Left, held-out IMPROVE R²
 * against TOR EC per k (median and IQR over the splits); right, the same splits as one point per k: that R²
 * against the All-Addis Deming intercept, joined in k order, so a flat validation under a falling intercept
 * shows as a vertical run. Against TOR and an intercept in µg/m³: nothing here moves with the MAC.
 */
export function ImproveBlind({ data, provenance }: { data: Data; provenance: ReactNode }) {
  const lg = useLegend()
  const baselines = baselinesOf(data)
  const im = rows(data, 'improve_k').filter((r) => r.set === 'all' && num(r.R2) != null)
  const ad = rows(data, 'addis_k').filter((r) => r.cohort === 'standard' && r.set === 'All Addis')
  const ks = [...new Set(im.map((r) => num(r.k)!))].sort((a, b) => a - b)
  const shown = baselines.filter((b) => lg.show(b.label))

  const r2: SweepSeries[] = shown.map((b) => ({
    id: b.key, label: b.label, color: b.color, group: b.label,
    points: im.filter((r) => r.baseline === b.key).map((r) => ({ x: num(r.k) ?? 0, y: num(r.R2), lo: num(r.R2_q25), hi: num(r.R2_q75) })),
  }))
  // one point per k: (IMPROVE R², Addis intercept); the line is drawn below in k order, not x order
  const pairs = shown.map((b) => ({ b, points: ks.flatMap((k): SweepPoint[] => {
    const x = at(im, b.key, 'all', k, 'R2'), y = at(ad, b.key, 'All Addis', k, 'intercept')
    return x == null || y == null ? [] : [{ x, y, id: String(k), tip: [b.label, `k = ${k}`, `held-out IMPROVE R² ${f2(x)}`, `All-Addis intercept ${f2(y)} µg/m³`] }]
  }) }))
  const scatter: SweepSeries[] = pairs.map(({ b, points }) => ({ id: b.key, label: b.label, color: b.color, group: b.label, points, width: 0, r: 2.6 }))
  const chosenR2 = shown.flatMap((b) => { const k = protocolK(data, b.key, 'standard'); return k != null ? [{ series: b.key, x: k }] : [] })
  const chosenPair = pairs.flatMap(({ b, points }) => {
    const k = protocolK(data, b.key, 'standard'), p = points.find((q) => q.id === String(k))
    return p ? [{ series: b.key, x: p.x }] : []
  })

  // the headline: how far R² and the intercept move between k = 11 and the last k
  const hi = ks[ks.length - 1], lo = ks.includes(11) ? 11 : ks[Math.floor(ks.length / 2)]
  const move = baselines.map((b) => ({ b, dR2: (at(im, b.key, 'all', hi, 'R2') ?? NaN) - (at(im, b.key, 'all', lo, 'R2') ?? NaN),
    dI: (at(ad, b.key, 'All Addis', hi, 'intercept') ?? NaN) - (at(ad, b.key, 'All Addis', lo, 'intercept') ?? NaN) })).filter((x) => Number.isFinite(x.dR2) && Number.isFinite(x.dI))
  const maxR2 = Math.max(...move.map((x) => Math.abs(x.dR2))), maxI = move.reduce((a, x) => (Math.abs(x.dI) > Math.abs(a.dI) ? x : a), move[0])

  return (
    <ChartFrame id="ksweep-improve-r2"
      title={`IMPROVE validation is blind to k — from k = ${lo} to ${hi} held-out R² moves by at most ${f2(maxR2)} while the All-Addis intercept moves ${f2(maxI?.dI)} µg/m³ (${maxI?.b.label})`}
      subtitle={`Notebook [5], standard all-IMPROVE lot-251 model. Left: held-out IMPROVE R² against TOR EC (OLS, held-out sites only), median and IQR over 100 site-disjoint splits; it plateaus by k ≈ 8–11. Right: the same split medians as one point per k, R² against the All-Addis Deming intercept, joined in k order (k = ${ks[0]} at the left). Ringed: the k the frozen protocol picks (1-SE rule on site-grouped CV). Validation that could see the Addis drift would move right as the intercept falls; instead the lines drop almost vertically. R² is against TOR and the intercept is in µg/m³, so neither moves with the MAC.`}
      provenance={provenance}>
      <Grid min={320}>
        <SweepPlot series={r2} chosen={chosenR2} xLabel="PLS components k" yLabel="Held-out IMPROVE R² (vs TOR)" height={280} title="IMPROVE validation" dim={lg.dim} />
        <SweepPlot series={scatter} chosen={chosenPair} xLabel="Held-out IMPROVE R² (vs TOR)" yLabel="All-Addis intercept (µg/m³)" reference={0} height={280}
          title="Same splits: validation against the Addis intercept" dim={lg.dim}>
          {({ x, y }) => <g>{pairs.map(({ b, points }) => (
            <path key={b.key} fill="none" stroke={b.color} strokeWidth={1.5} opacity={lg.dim(b.label)}
              d={d3.line<SweepPoint>().x((p) => (x as d3.ScaleLinear<number, number>)(p.x as number)).y((p) => y(p.y as number))(points) ?? ''} />))}</g>}
        </SweepPlot>
      </Grid>
      <Legend {...lg.props} items={baselines.map((b) => ({ label: b.label, color: b.color, shape: 'line' as const }))}
        note={im.some((r) => num(r.R2_q25) != null) ? 'median and IQR over 100 splits' : 'median over 100 splits, the seven k the notebook displayed'} />
    </ChartFrame>
  )
}
