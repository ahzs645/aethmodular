import { useMemo } from 'react'
import { ChartFrame, Empty, Note, Segmented } from '@/components/ChartFrame'
import { Key } from '@/components/Key'
import { SweepPlot, type SweepMark, type SweepSeries } from '@/components/SweepPlot'
import { fmt } from '@/lib/format'
import { useMac } from '@/lib/mac'
import { useHighlightActions } from '@/lib/highlight'
import { ExcludedNote, useMarkReason } from '@/lib/markedRows'
import { markTip } from '@/lib/exclusionMode'
import { seasonOnly } from '@/lib/seasons'
import { useSearchParam } from '@/lib/url'
import { FONT, INK } from '@/lib/theme'
import type { MetaFile } from '@/lib/types'
import { MAC_DEFS, bcTag, macMarker, dateDomain, dateTicks, macOf, seasonColor, seasonsIn, type Channel, type MacKey, type Pair, type PairSite } from './data'
import { iqrFences, seasonRuns } from './mac'

const GAP_DAYS = 60, MIN_RUN = 5

/**
 * Per-filter MAC through time, coloured by season, with the average of each run of one season (the
 * diamonds of AAARpos and ETAD_Aug13): a season's filters split wherever two are more than 60 days
 * apart, runs of at least 5 averaged and labelled. The notebooks dropped IQR outliers first; here they
 * stay on the chart, hollow, and are left out of the averages.
 */
export function MacTimeline({ pairs, marked = [], meta, sites, ch, sigma }: {
  pairs: Pair[]; meta: MetaFile; sites: PairSite[]; ch: Channel; sigma: number | null
  /** Filters → excluded: Mark, the excluded filters paired the same way (lib/markedRows): ×, in no fence or run mean */
  marked?: Pair[]
}) {
  const [key, setKey] = useSearchParam<MacKey>('af.mac', 'hipsOverEc', MAC_DEFS.map((d) => d.key))
  const def = MAC_DEFS.find((d) => d.key === key)!
  const hl = useHighlightActions()
  const reasonOf = useMarkReason()
  const per = useMemo(() => sites.map((s) => {
    const pts = pairs.filter((p) => p.site === s.name).flatMap((p) => { const v = macOf(def, p); return v === null ? [] : [{ p, v }] })
    const fences = iqrFences(pts.map((q) => q.v))
    const isOut = (v: number) => !!fences && (v < fences[0] || v > fences[1])
    const kept = pts.filter((q) => !isOut(q.v))
    const runs = seasonRuns(kept, (q) => q.p.season, (q) => q.p.date, (q) => q.v, { gapDays: GAP_DAYS, minN: MIN_RUN })
    const series: SweepSeries[] = seasonsIn(pts.map((q) => q.p), meta).map((se) => ({
      id: `${s.code}|${se}`, label: seasonOnly(se), color: seasonColor(meta, se, INK.muted), width: 0, marks: true, r: 3,
      points: pts.filter((q) => q.p.season === se && !isOut(q.v)).map((q) => ({ x: new Date(q.p.date), y: q.v, id: q.p.id,
        tip: [`${q.p.id} · ${q.p.date}`, `${def.label}: ${fmt(q.v, 2)} m²/g`, seasonOnly(se), 'click to open the filter'] })),
    }))
    const outliers = pts.filter((q) => isOut(q.v))
    const marks: SweepMark[] = marked.filter((p) => p.site === s.name).flatMap((p) => { const v = macOf(def, p); return v === null ? [] : [{ x: new Date(p.date), y: v, tip: markTip(p, reasonOf(p.id) ?? 'excluded', [`${def.label}: ${fmt(v, 2)} m²/g`]) }] })
    return { s, n: pts.length, series, runs, outliers, marks, fences, ticks: dateTicks(pts.map((q) => q.p.date)), xDomain: dateDomain(pts.map((q) => q.p.date)) }
  }).filter((x) => x.n > 0), [pairs, marked, meta, sites, def, reasonOf])

  const reference = def.reference(sigma ?? NaN)
  const { mac } = useMac()
  const readerMac = macMarker(def, mac)
  return (
    <ChartFrame id="aeth-filters-mac-time" title="MAC through time — each filter, and the average of each season's run"
      subtitle={`${def.label} per filter (m²/g), coloured by season under the gallery's calendar. Diamonds: the mean of each run of one season (filters no more than ${GAP_DAYS} days apart, at least ${MIN_RUN}), at the run's middle filter, as in the notebooks. Filters outside 1.5 × IQR of the site's values (the notebooks' remove_outliers_iqr) are drawn hollow and left out of the averages. The line is ${Number.isFinite(reference) ? `${fmt(reference, 1)} m²/g, ${def.refLabel}` : 'absent until σ_ATN is loaded'}${readerMac ? `; dashed, the MAC set under Filters → units (${mac})` : ''}.`}
      provenance="AAARpos.ipynb (MAC time series by season, instance averages) · ETAD_Aug13.ipynb §9.6 · ETAD_comprehensive_absorption_analysis.ipynb §10.5–11 (aethalometer MAC with seasonal instance averages)"
      controls={<Segmented label="MAC" value={key} options={MAC_DEFS.map((d) => d.key)} onChange={setKey}
        optionLabel={(k) => { const d = MAC_DEFS.find((x) => x.key === k)!; return k === 'hipsOverEc' ? d.short : k === 'hipsOverBc' ? `HIPS ÷ ${ch} BC` : `b_ATN ${ch} ÷ EC` }} />}>
      {per.length === 0 ? <Empty>No filter in view has both members of {def.label}.</Empty> : per.map(({ s, series, runs, outliers, marks, ticks, xDomain }) => (
        <div key={s.code}>
          <SweepPlot series={series} xScale="time" xDomain={xDomain} xFormat={ticks} yLabel={`${def.short}${def.key === 'hipsOverEc' ? '' : bcTag(pairs.filter((p) => p.site === s.name))} (m²/g)`} height={260} reference={Number.isFinite(reference) ? reference : undefined}
            yMarkers={readerMac ? [{ at: readerMac.value, label: readerMac.label, color: INK.accent }] : undefined}
            title={sites.length > 1 ? s.name : undefined}
            onPoint={(_, pt) => pt.id && hl.openSample(pt.id)} marks={marks}
            children={({ x, y }) => <g>
              {/* outliers do not stretch the axis: one beyond it sits on the edge */}
              {outliers.map((q) => {
                const [a, b] = y.domain(), v = Math.min(Math.max(q.v, Math.min(a, b)), Math.max(a, b))
                return <circle key={q.p.id} cx={(x as (d: Date) => number)(new Date(q.p.date))} cy={y(v)} r={3.2} fill="none"
                  stroke={seasonColor(meta, q.p.season, INK.muted)} strokeWidth={1.2} />
              })}
              {runs.map((r, i) => {
                const cx = (x as (d: Date) => number)(new Date(r.date)), cy = y(r.mean), c = seasonColor(meta, r.season, INK.muted)
                return <g key={i}>
                  <path d={`M${cx} ${cy - 7} L${cx + 7} ${cy} L${cx} ${cy + 7} L${cx - 7} ${cy} Z`} fill={INK.panel} stroke={c} strokeWidth={2} />
                  <text x={cx} y={cy - 11} textAnchor="middle" fontSize={10.5} fontFamily={FONT.mono} fill={INK.text} paintOrder="stroke" stroke={INK.panel} strokeWidth={3}>{fmt(r.mean, 1)}</text>
                </g>
              })}
            </g>} />
          {outliers.length > 0 && <Note explain>{s.name}: {outliers.length} filter{outliers.length > 1 ? 's' : ''} outside 1.5 × IQR ({outliers.map((q) => `${q.p.date} ${fmt(q.v, 1)}`).join(', ')}), drawn hollow (on the axis edge when beyond it) and left out of the averages.</Note>}
        </div>
      ))}
      <ExcludedNote count={per.reduce((a, x) => a + x.marks.length, 0)} drawn />
      <Key items={[...meta.seasons.filter((se) => pairs.some((p) => p.season === se.name)).map((se) => ({ label: seasonOnly(se.name), color: se.color })),
        { label: 'season run mean', color: INK.muted, kind: 'diamond' }, { label: 'outside 1.5 × IQR', color: INK.muted, kind: 'hollow' }]} />
    </ChartFrame>
  )
}
