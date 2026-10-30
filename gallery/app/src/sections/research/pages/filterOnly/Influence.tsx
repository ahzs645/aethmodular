/**
 * One filter's influence (stability 04 and 09): the release asks it of ETAD-0037, the Addis filter it keeps in the
 * baseline and the exclusion registry; here any diagnostic filter can be the one left out.
 */
import { useMemo, type ReactNode } from 'react'
import type { ScalePoint } from 'd3'
import { ChartFrame, Empty, Note, Segmented, Select } from '@/components/ChartFrame'
import { IntervalStrip, type IntervalRow } from '@/components/IntervalStrip'
import { SweepPlot, type SweepSeries } from '@/components/SweepPlot'
import { fmt, signed } from '@/lib/format'
import { INK } from '@/lib/theme'
import type { MetaFile } from '@/lib/types'
import { useSearchParam } from '@/lib/url'
import { evaluate, fitTraining, inPopulation, pairedInfluence, type FPoint, type Scheme } from '../../common/filterOnly'
import { NB_STAB, SCHEMES, SOURCE, barLayer, popLabel, schemeLabel, shortQ, siteColor, type Settings } from './data'

/** the filter left out, in the link as fo.omit (the release's ETAD-0037 by default) */
export function useOmitted(points: FPoint[], fallback: string): [FPoint | null, (id: string) => void, string[]] {
  const ids = useMemo(() => points.map((p) => p.id).sort(), [points])
  const [id, setId] = useSearchParam<string>('fo.omit', fallback)
  return [points.find((p) => p.id === id) ?? points.find((p) => p.id === fallback) ?? null, setId, ids]
}

function OmitPick({ value, ids, points, onChange }: { value: string; ids: string[]; points: FPoint[]; onChange: (v: string) => void }) {
  const site = new Map(points.map((p) => [p.id, p.site]))
  return <Select label="Filter left out" value={value} options={ids} onChange={onChange} optionLabel={(o) => `${o} · ${site.get(o) ?? ''}`} />
}

interface Influence { site: string; cohort: FPoint[]; inCohort: boolean; full: ReturnType<typeof fitTraining>; without: ReturnType<typeof fitTraining>; rank: number | null }

function useInfluence(points: FPoint[], omitted: FPoint | null, s: Settings): Influence | null {
  return useMemo(() => {
    if (!omitted) return null
    const cohort = points.filter((p) => p.site === omitted.site && inPopulation(p, s.population))
    const full = fitTraining(cohort, s.minTrain)
    const without = fitTraining(cohort.filter((p) => p.id !== omitted.id), s.minTrain)
    // where the chosen filter ranks among the site's filters by how far leaving it out moves the slope
    const moves = cohort.map((f) => ({ id: f.id, d: Math.abs(fitTraining(cohort.filter((p) => p !== f), s.minTrain).slope - full.slope) })).sort((a, b) => b.d - a.d)
    const at = moves.findIndex((m) => m.id === omitted.id)
    return { site: omitted.site, cohort, inCohort: cohort.some((p) => p.id === omitted.id), full, without, rank: at >= 0 ? at + 1 : null }
  }, [points, omitted, s.population, s.minTrain])
}

/** stability 04: coefficient changes from leaving one filter out against leaving a whole quarter out */
export function CoefficientInfluence({ points, s, meta, omitDefault }: { points: FPoint[]; s: Settings; meta: MetaFile; omitDefault: string }) {
  const [omitted, setOmit, ids] = useOmitted(points, omitDefault)
  const inf = useInfluence(points, omitted, s)
  const controls = <OmitPick value={omitted?.id ?? ''} ids={ids} points={points} onChange={setOmit} />
  const frame = (body: ReactNode) => (
    <ChartFrame id="fo-coef" title="One filter against whole quarters — how far one omission moves the OLS coefficients"
      subtitle={`${inf?.site ?? ''} · ${popLabel(s.population)} population · change from the full-cohort fit: leaving out ${omitted?.id ?? 'a filter'}, or each withheld quarter's training fit`}
      provenance={<>{NB_STAB} figure 04 (ETAD-0037)</>} source={`${SOURCE}/Influence.tsx › CoefficientInfluence`}
      tip={<>The full-cohort OLS coefficients are descriptive references, not validation scores. At the release's settings, leaving out ETAD-0037 changes the Addis slope by +0.033 and the intercept by −0.206 Mm⁻¹; whole-quarter omissions move them further. The filter stays in the baseline and in the exclusion registry.</>}
      controls={controls}>{body}</ChartFrame>
  )
  if (!inf || !omitted) return frame(<Empty>No filter to leave out.</Empty>)
  if (!inf.inCohort) return frame(<Empty>{omitted.id} is not in the {popLabel(s.population)} population at {inf.site}; pick another filter or population.</Empty>)
  if (inf.full.status !== 'evaluated' || inf.without.status !== 'evaluated') return frame(<Empty>{inf.site}'s {popLabel(s.population)} population has too few filters for a fit.</Empty>)
  const color = siteColor(meta, inf.site)
  const quarters = evaluate(points.filter((p) => p.site === inf.site), inf.cohort, 'leave_quarter_out', s.minTrain).filter((b) => b.testN > 0)
  const bar = (v: number, c: string, faint = false) => ({ mid: v, lo: Math.min(0, v), hi: Math.max(0, v), color: c, faint, shape: 'square' as const, r: 3.5 })
  const ds = inf.without.slope - inf.full.slope, di = (inf.without.intercept - inf.full.intercept) * s.k
  const rows: IntervalRow[] = [
    { key: 'one', label: omitted.id, strong: true, labelColor: color, marks: [[bar(ds * s.k, color)], [bar(di, color)]], note: `rank ${inf.rank}/${inf.cohort.length}`,
      tip: [`Leave out ${omitted.id} (${omitted.date})`, `slope ${signed(ds * s.k, 3)} · intercept ${signed(di, 3)} ${s.u}`, `ranks ${inf.rank} of ${inf.cohort.length} filters by |slope change|`] },
    ...quarters.map((b) => {
      const dsq = b.status === 'evaluated' ? (b.fit.slope - inf.full.slope) * s.k : null, diq = b.status === 'evaluated' ? (b.fit.intercept - inf.full.intercept) * s.k : null
      return { key: b.block, label: `without ${b.block}`, note: `n=${b.testN}`, marks: [dsq === null ? [] : [bar(dsq, '#8a8a8a')], diq === null ? [] : [bar(diq, '#8a8a8a')]],
        tip: [`Leave out ${b.block} (${b.testN} filters)`, dsq === null ? 'no fit' : `slope ${signed(dsq, 3)} · intercept ${signed(diq, 3)} ${s.u}`] }
    }),
  ]
  return frame(<>
    <IntervalStrip blocks={[{ key: 'rows', rows }]} labelWidth={130} noteWidth={92} rowHeight={22}
      panels={[{ label: `Change in OLS slope (${s.u} per µg/m³)`, reference: 0, format: (v) => fmt(v, 2) }, { label: `Change in OLS intercept (${s.u})`, reference: 0, format: (v) => fmt(v, 1) }]} />
    <Note>Full-cohort fit at {inf.site}: slope {fmt(inf.full.slope * s.k, 3)}, intercept {fmt(inf.full.intercept * s.k, 3)} {s.u}, n {inf.cohort.length}. The rank compares leaving out {omitted.id} with leaving out each of the site's other filters in turn.</Note>
  </>)
}

/** stability 09: MAE on the same withheld filters with and without the left-out filter in training */
export function CommonFilterInfluence({ points, s, meta, omitDefault }: { points: FPoint[]; s: Settings; meta: MetaFile; omitDefault: string }) {
  const [omitted, setOmit, ids] = useOmitted(points, omitDefault)
  const [scheme, setScheme] = useSearchParam<Scheme>('fo.common.split', 'leave_quarter_out', SCHEMES)
  const rows = useMemo(() => {
    if (!omitted) return []
    const diag = points.filter((p) => p.site === omitted.site), cohort = diag.filter((p) => inPopulation(p, s.population))
    return pairedInfluence(evaluate(diag, cohort, scheme, s.minTrain), evaluate(diag, cohort.filter((p) => p.id !== omitted.id), scheme, s.minTrain))
  }, [points, omitted, s.population, s.minTrain, scheme])
  const color = omitted ? siteColor(meta, omitted.site) : INK.accent
  const series: SweepSeries[] = [{ id: 'd', label: 'MAE change', color, width: 0, r: 0, points: rows.map((r) => ({ x: r.block, y: r.change * s.k,
    tip: [r.block, `${r.n} common filters`, `MAE ${fmt(r.baseMae * s.k, 3)} → ${fmt(r.withoutMae * s.k, 3)} ${s.u}`, `change ${signed(r.change * s.k, 4)}`] })) }]
  const all = rows[0]
  return (
    <ChartFrame id="fo-common" title="Errors on common filters — what leaving one filter out of training does to the others' predictions"
      subtitle={`${omitted?.site ?? ''} · ${popLabel(s.population)} population · ${schemeLabel(scheme).toLowerCase()} · MAE after omitting ${omitted?.id ?? 'the filter'} minus MAE with it, on filters predicted both ways; negative = lower error after omission`}
      provenance={<>{NB_STAB} figure 09 (ETAD-0037)</>} source={`${SOURCE}/Influence.tsx › CommonFilterInfluence`}
      tip={<>At the release's settings the ETAD-0037 comparison covers the same 189 other filters; leave-quarter-out MAE changes by only −0.0049 Mm⁻¹. Predictions in its own withheld quarter do not change, because it was already absent from that training. Later-period common-filter MAE worsens by +0.172 Mm⁻¹ after omission, mostly in the first evaluated later quarter.</>}
      controls={<><OmitPick value={omitted?.id ?? ''} ids={ids} points={points} onChange={setOmit} /><Segmented label="Split" value={scheme} options={SCHEMES} onChange={setScheme} optionLabel={schemeLabel} /></>}>
      {rows.length
        ? <SweepPlot series={series} xScale="point" height={250} xFormat={(v) => (v === 'All common filters' ? 'All' : shortQ(v))} yLabel={`MAE change after omission (${s.u})`} reference={0}
            background={barLayer(rows.map((r, i) => ({ x: r.block, y: r.change * s.k, color, faint: i > 0 })))}
            children={({ x, ih }) => <g>{rows.map((r) => <text key={r.block} x={(x as ScalePoint<string>)(r.block) ?? 0} y={ih - 4} textAnchor="middle" fontSize={9} fill={INK.muted}>n={r.n}</text>)}</g>} />
        : <Empty>No filters are predicted under both trainings here.</Empty>}
      {all && <Note>All {all.n} common filters: MAE {fmt(all.baseMae * s.k, 3)} → {fmt(all.withoutMae * s.k, 3)} {s.u} ({signed(all.change * s.k, 4)}). The first bar pools every quarter; the paler bars are the quarters.</Note>}
    </ChartFrame>
  )
}
