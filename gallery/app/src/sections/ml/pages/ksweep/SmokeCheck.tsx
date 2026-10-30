import type { ReactNode } from 'react'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { Legend, useLegend } from '@/components/Legend'
import { SweepPlot, type SweepSeries } from '@/components/SweepPlot'
import { useChartOptions } from '@/lib/chartOptions'
import { f2 } from '@/lib/format'
import { useMac } from '@/lib/mac'
import { shortSeason } from '@/lib/seasons'
import { INK } from '@/lib/theme'
import { num, type Row } from '@/lib/table'
import { rows } from '../../common'
import { Grid } from '../../../research/common/ui'
import { GROUPS, SEASON_COLOR, SUBSET_COLOR, SUBSET_SHORT, baselinesOf, sharedDomain, type Data } from './common'
import { FixedSetNote } from '@/lib/markedRows'

const ADDIS = 'Addis (vs Fabs/MAC)'

/**
 * Notebook [7] (split medians) and [9] (pooled out-of-fold): the deciding check. Do smoke-like IMPROVE filters
 * drift with k the way Addis does? Intercepts are divided by each population's mean reference, so Addis
 * (about 4.7 µg/m³ at MAC 10) and IMPROVE (0.1–0.5) share an axis; OLS on both sides. Three panels for one
 * baseline: Addis against HIPS, held-out IMPROVE against TOR EC, held-out IMPROVE against IMPROVE HIPS.
 *
 * The MAC: an intercept ÷ mean Fabs/MAC scales by MAC/10 (the intercept stays, the mean HIPS BC goes as 1/MAC),
 * so the Addis and IMPROVE-vs-HIPS curves and the reference lines follow the reader's MAC; IMPROVE vs TOR does not.
 * The dashed lines are each subset's TOR EC vs HIPS line (no FTIR): where a model that reproduced TOR exactly
 * would land. By default they are divided by the subset's mean HIPS, the curves' own denominator; the notebook
 * divided them by mean TOR (its reference_gap intercept_rel), which a choice restores.
 */
export function SmokeCheck({ data, provenance }: { data: Data; provenance: ReactNode }) {
  const m = useMac()
  const lg = useLegend()
  const baselines = baselinesOf(data)
  const improve = rows(data, 'improve_k'), pooled = rows(data, 'pooled'), addisRel = rows(data, 'addis_rel')
  const gap = rows(data, 'reference_gap')
  const subsets = rows(data, 'subsets')
  const has = (rs: Row[], col: string, b: string) => rs.some((r) => r.baseline === b && num(r[col]) != null)
  const offered = baselines.filter((b) => has(improve, 'hips_intercept_rel', b.key) || has(pooled, 'rel_hips', b.key))
  const o = useChartOptions('ksweep-smoke', [
    { key: 'baseline', label: 'Baseline', default: 'AIRSpec', style: 'select', choices: offered.map((b) => ({ value: b.key, label: b.label })) },
    { key: 'estimate', label: 'IMPROVE estimate', default: 'splits',
      choices: [{ value: 'splits', label: 'Median of 100 splits' }, { value: 'pooled', label: 'Pooled out-of-fold' }] },
    { key: 'gap', label: 'Dashed TOR-vs-HIPS line', default: 'hips',
      choices: [{ value: 'hips', label: '÷ mean HIPS (as the curves)' }, { value: 'tor', label: '÷ mean TOR (as the notebook)' }] },
  ])
  const b = offered.find((x) => x.key === o.values.baseline) ?? offered[0]
  const pooledMode = o.values.estimate === 'pooled'
  const atMac = (v: number | null) => m.slope(v)  // ÷ mean Fabs/MAC: × MAC/10 like a slope
  if (!b) return null

  const ks = (rs: Row[]) => [...rs].sort((p, q) => (num(p.k) ?? 0) - (num(q.k) ?? 0))
  const label = (s: string) => SUBSET_SHORT[s] ?? s
  const n = (s: string) => num(subsets.find((r) => r.set === s)?.n)
  const subsetNames = subsets.map((r) => String(r.set))

  // Addis: OLS intercept ÷ mean Fabs/MAC, split medians (Addis is predicted in full by every split, so there is no pooled version)
  const addisLine = (g: string, width: number, faint = false): SweepSeries | null => {
    const rs = ks(addisRel.filter((r) => r.baseline === b.key && r.set === g))
    if (!rs.length) return null
    const name = g === 'All Addis' ? ADDIS : shortSeason(g)
    return { id: `addis-${g}${faint ? '-faint' : ''}`, label: name, group: name, color: g === 'All Addis' ? INK.text : SEASON_COLOR[g] ?? INK.muted, width, marks: false,
      fill: faint ? 0 : 0.1, interval: faint ? 'none' : 'band',
      points: rs.map((r) => ({ x: num(r.k) ?? 0, y: atMac(num(r.rel)), lo: atMac(num(r.rel_q25)), hi: atMac(num(r.rel_q75)),
        tip: [`Addis · ${shortSeason(g)}`, `k = ${r.k}`, `OLS intercept ÷ mean Fabs/${m.mac}: ${f2(atMac(num(r.rel)))}`] })) }
  }
  const addisPanel = GROUPS.map((g) => addisLine(g, g === 'All Addis' ? 2.6 : 1.3)).filter((s): s is SweepSeries => !!s && lg.show(s.label))
  const scale = addisLine('All Addis', 2.4, true)
  const forScale = scale && lg.show(ADDIS) ? [{ ...scale, color: INK.muted }] : []

  // IMPROVE: one line per subset, against TOR (MAC-free) or IMPROVE HIPS Fabs/MAC
  const improveLines = (ref: 'tor' | 'hips'): SweepSeries[] => subsetNames.filter((s) => lg.show(label(s))).flatMap((s) => {
    const src = pooledMode ? pooled : improve
    const col = pooledMode ? (ref === 'tor' ? 'rel_tor' : 'rel_hips') : ref === 'tor' ? 'intercept_rel' : 'hips_intercept_rel'
    const rs = ks(src.filter((r) => r.baseline === b.key && r.set === s && num(r[col]) != null))
    if (!rs.length) return []
    const v = (c: string) => (r: Row) => (ref === 'hips' ? atMac(num(r[c])) : num(r[c]))
    return [{ id: `${ref}-${s}`, label: label(s), group: label(s), color: SUBSET_COLOR[s] ?? INK.axis, width: 1.7, marks: rs.length <= 8,
      points: rs.map((r) => ({ x: num(r.k) ?? 0, y: v(col)(r), lo: pooledMode ? null : v(`${col}_q25`)(r), hi: pooledMode ? null : v(`${col}_q75`)(r),
        tip: [`${label(s)} · ${b.label}`, `k = ${r.k}`, `intercept ÷ mean ${ref === 'tor' ? 'TOR EC' : `HIPS Fabs/${m.mac}`}: ${f2(v(col)(r))}`] })) }]
  })
  const tor = improveLines('tor'), hips = improveLines('hips')
  const gapAt = (s: string) => {
    const r = gap.find((x) => x.set === s)
    if (!r) return null
    return o.values.gap === 'tor' ? num(r.intercept_rel) : atMac((num(r.intercept) ?? NaN) / (num(r.x_mean) ?? NaN))
  }
  const sg = gap.find((x) => x.set === 'smoke_cohort')
  const smokeGap = { tor: num(sg?.intercept_rel ?? null), hips: atMac((num(sg?.intercept ?? null) ?? NaN) / (num(sg?.x_mean ?? null) ?? NaN)) }
  const markers = subsetNames.filter((s) => lg.show(label(s))).flatMap((s) => { const v = gapAt(s); return v != null && Number.isFinite(v) ? [{ at: v, color: SUBSET_COLOR[s] }] : [] })

  const panels = [
    { key: 'addis', title: `${b.label}: Addis`, series: addisPanel, yLabel: `OLS intercept ÷ mean Fabs/${m.mac}` },
    { key: 'tor', title: `${b.label}: held-out IMPROVE vs TOR EC`, series: [...forScale, ...tor], yLabel: 'OLS intercept ÷ mean TOR EC' },
    { key: 'hips', title: `${b.label}: held-out IMPROVE vs HIPS Fabs/${m.mac}`, series: [...forScale, ...hips], yLabel: `OLS intercept ÷ mean Fabs/${m.mac}`, markers },
  ]
  const yDomain = sharedDomain([...panels.flatMap((p) => p.series.flatMap((s) => s.points.flatMap((q) => [q.y, q.lo, q.hi]))), ...markers.map((mk) => mk.at)])

  // the headline, read off the drawn lines at the last k
  const last = (s: SweepSeries | undefined) => { const p = s?.points[s.points.length - 1]; return p ? { k: p.x as number, y: p.y ?? null } : null }
  const addisEnd = last(addisPanel.find((s) => s.label === ADDIS) ?? addisLine('All Addis', 2) ?? undefined)
  const smokeEnds = hips.filter((s) => !s.id.endsWith('-all') && !s.id.endsWith('nsK_bottom50')).map((s) => last(s)?.y).filter((v): v is number => v != null)
  const title = addisEnd && smokeEnds.length
    ? `Do smoke-like IMPROVE filters drift like Addis? No — at k = ${addisEnd.k} Addis (${b.label}) sits at ${f2(addisEnd.y)} of its mean, the smoke-like subsets at ${f2(Math.min(...smokeEnds))} to ${f2(Math.max(...smokeEnds))} of theirs (${m.label})`
    : 'Do smoke-like IMPROVE filters drift like Addis?'
  const empty = (p: (typeof panels)[number]) => p.series.filter((s) => !s.id.endsWith('-faint')).length === 0

  return (
    <ChartFrame id="ksweep-smoke" title={title}
      subtitle={`Notebooks [7] and [9], standard all-IMPROVE lot-251 model. The populations differ about 10× in concentration, so every intercept is divided by its population's mean reference, with OLS on both sides (the Addis Deming line is in the first chart). Subsets are fixed from IMPROVE data alone before any Addis readout: non-soil K = K − 0.6 Fe ≥ the library p90 (n = ${n('nsK_top10') ?? '?'}), TOR OC/EC ≥ p90 (n = ${n('ocec_top10') ?? '?'}), the phase-2 smoke cohort (n = ${n('smoke_cohort') ?? '?'}), against non-soil K ≤ the median (n = ${n('nsK_bottom50') ?? '?'}) and all ${n('all')?.toLocaleString() ?? ''} lot-251 filters; each curve uses the held-out sites of each split. ${pooledMode ? "Pooled: each filter's prediction averaged over the splits that held its site out, one line per subset ([9]); the Addis lines stay split medians." : 'Split medians with the IQR ([7]).'} The thick grey line in the IMPROVE panels repeats All-Addis for scale. Dashed: each subset's own TOR EC vs HIPS line (OLS, no FTIR), ${o.values.gap === 'tor' ? "divided by the subset's mean TOR as the notebook drew it" : `divided by the subset's mean HIPS like the curves (the notebook divided by mean TOR, which puts the smoke cohort's at ${f2(smokeGap.tor)} instead of ${f2(smokeGap.hips)})`}. Against HIPS the smoke-like subsets level off by k ≈ 7 for spline and VIBES alike; the Addis VIBES curves keep falling to k ≈ 18. ${m.changed ? `Everything divided by mean Fabs/MAC is × ${m.mac}/10 of the notebook's MAC 10 numbers; the TOR panel does not move.` : 'MAC 10: the curves divided by mean Fabs/MAC follow the MAC set under Filters → units; the TOR panel does not.'}`}
      provenance={provenance} controls={o.controls()}>
      <Grid min={300}>
        {panels.map((p) => <div key={p.key}>{empty(p)
          ? <><div className="facet-title">{p.title}</div><Empty>The notebook displayed this readout only for the IMPROVE-vs-HIPS split medians; the per-split table (curves.csv) adds it.</Empty></>
          : <SweepPlot series={p.series} title={p.title} xLabel="PLS components k" yLabel={p.yLabel} reference={0} height={270} yDomain={yDomain} dim={lg.dim}
            yMarkers={p.markers?.map((mk) => ({ at: mk.at, color: mk.color }))} />}</div>)}
      </Grid>
      <Legend {...lg.props} items={[
        ...addisPanel.length ? [{ label: ADDIS, color: INK.text, shape: 'line' as const }] : [],
        ...GROUPS.filter((g) => g !== 'All Addis' && addisRel.some((r) => r.set === g)).map((g) => ({ label: shortSeason(g), color: SEASON_COLOR[g], shape: 'line' as const })),
        ...subsetNames.map((s) => ({ label: label(s), color: SUBSET_COLOR[s] ?? INK.axis, shape: 'line' as const })),
      ]} note="dashed: the subset's TOR-vs-HIPS line" />
      <FixedSetNote what="the notebook's saved curves on IMPROVE subsets and its own Addis filters" />
    </ChartFrame>
  )
}
