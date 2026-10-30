import { useMemo } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { Key } from '@/components/Key'
import { SweepPlot, type SweepSeries } from '@/components/SweepPlot'
import { fmt } from '@/lib/format'
import { useMac } from '@/lib/mac'
import { useHighlightActions } from '@/lib/highlight'
import { ExcludedNote } from '@/lib/markedRows'
import { networkColor } from '@/lib/networks'
import { splitAtGaps } from '@/lib/stats'
import { useSearchParam } from '@/lib/url'
import { dateDomain, dateTicks, type Channel, type Pair, type PairSite } from './data'
import { SweepMarks, type SweepMark } from '../sweepMarks'

const SCALES = ['absolute', 'median'] as const
const MEASURES = (ch: Channel, mac: number) => [
  { key: 'aeth', label: `Aethalometer ${ch} BCc`, value: (p: Pair) => p.bc },
  { key: 'hips', label: `HIPS BC (Fabs ÷ ${mac})`, value: (p: Pair) => p.hipsBc },
  { key: 'ec', label: 'FTIR EC', value: (p: Pair) => p.ec },
]

/**
 * The three methods filter by filter through time (AAARpos's and ETAD_Aug13 §11's method-agreement
 * series), in µg/m³ or each divided by its own median over the site's filters so their timing can be
 * compared (ETAD_Aug13 §9 normalised by the maximum instead, which one high filter decides).
 */
export function AgreementTimeline({ pairs, marked = [], sites, ch }: {
  pairs: Pair[]; sites: PairSite[]; ch: Channel
  /** Filters → excluded: Mark, the excluded filters paired the same way (lib/markedRows): ×, in no median or mean */
  marked?: Pair[]
}) {
  const [scale, setScale] = useSearchParam<(typeof SCALES)[number]>('af.norm', 'absolute', SCALES)
  const hl = useHighlightActions()
  const { mac } = useMac()
  const measures = MEASURES(ch, mac)
  const per = useMemo(() => sites.map((s) => {
    const own = pairs.filter((p) => p.site === s.name).sort((a, b) => a.date.localeCompare(b.date))
    const all = own.filter((p) => measures.every((m) => m.value(p) !== null))
    const markedOwn = marked.filter((p) => p.site === s.name)
    const marks: SweepMark[] = []
    const series: SweepSeries[] = measures.flatMap((m, i) => {
      const pts = own.filter((p) => m.value(p) !== null)
      const med = d3.median(pts, m.value) ?? 1
      const div = scale === 'median' && med > 0 ? med : 1
      // the excluded filters on the counted filters' scale (their median), each method its own ×
      for (const p of markedOwn) { const v = m.value(p); if (v !== null) marks.push({ x: new Date(p.date), y: v / div }) }
      return splitAtGaps(pts, (p) => new Date(p.date)).map((run, j) => ({
        id: `${m.key}${j}`, label: m.label, group: m.key, color: networkColor(i), marks: true, r: 2.4, width: 1.4,
        points: run.map((p) => ({ x: new Date(p.date), y: m.value(p)! / div, id: p.id,
          tip: [`${p.id} · ${p.date}`, `${m.label}: ${fmt(m.value(p), 2)} µg/m³${div !== 1 ? ` (${fmt(m.value(p)! / div, 2)} × its median)` : ''}`, 'click to open the filter'] })),
      }))
    })
    const means = measures.map((m) => d3.mean(all, m.value))
    return { s, series, marks, nMarked: markedOwn.filter((p) => measures.some((m) => m.value(p) !== null)).length, n: own.length, nAll: all.length, means, ticks: dateTicks(own.map((p) => p.date)), xDomain: dateDomain(own.map((p) => p.date)) }
  }).filter((x) => x.n > 0), [pairs, marked, sites, scale, ch]) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <ChartFrame id="aeth-filters-timeline" title="Through time — the aethalometer, HIPS and FTIR EC on each filter"
      subtitle={`Each filter's aethalometer ${ch} BCc (window mean), HIPS BC (Fabs ÷ ${mac}, the MAC set under Filters → units; the AAARpos notebook used 10) and FTIR EC, in µg/m³ or each divided by its own median at the site. Lines break where filters are more than 45 days apart. The means over the filters with all three are under each site.`}
      provenance="AAARpos.ipynb (method agreement time series) · ETAD_Aug13.ipynb §9, §11 · ETAD_comprehensive_absorption_analysis.ipynb §9 (normalised FTIR EC vs BCc)"
      controls={<Segmented label="Scale" value={scale} options={SCALES} onChange={setScale} optionLabel={(k) => (k === 'absolute' ? 'µg/m³' : '÷ own median')} />}>
      {per.length === 0 ? <Empty>No filter in view has the aethalometer at {ch}.</Empty> : per.map(({ s, series, marks, nAll, means, ticks, xDomain }) => (
        <div key={s.code}>
          <SweepPlot series={series} xScale="time" xDomain={xDomain} xFormat={ticks} yLabel={scale === 'median' ? '÷ own median' : 'µg/m³'} height={240} include={[0]}
            title={sites.length > 1 ? s.name : undefined} reference={scale === 'median' ? 1 : undefined}
            onPoint={(_, pt) => pt.id && hl.openSample(pt.id)} children={({ x, y }) => <SweepMarks marks={marks} x={x} y={y} />} />
          <p className="chart-note">{nAll === 0 ? `${s.name}: no filter has all three.` : <>{s.name}, {nAll} filters with all three: mean {measures.map((m, i) => `${m.label} ${fmt(means[i], 2)}`).join(' · ')} µg/m³
            {means[0] && means[1] && means[2] ? ` · aethalometer ÷ mean of HIPS and EC ${fmt(means[0] / ((means[1] + means[2]) / 2), 2)} (ETAD_Aug13's "overestimation factor")` : ''}</>}</p>
        </div>
      ))}
      <ExcludedNote count={per.reduce((a, x) => a + x.nMarked, 0)} drawn />
      <Key items={measures.map((m, i) => ({ label: m.label, color: networkColor(i), kind: 'line' as const }))} />
    </ChartFrame>
  )
}
