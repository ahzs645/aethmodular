import { useMemo } from 'react'
import { ChartFrame, Empty } from '@/components/ChartFrame'
import { FacetGrid } from '@/components/FacetGrid'
import { Key } from '@/components/Key'
import { SweepPlot, type SweepSeries } from '@/components/SweepPlot'
import { f2 } from '@/lib/format'
import { splitAtGaps } from '@/lib/stats'
import { INK } from '@/lib/theme'
import { AAE_PAIRS, CLASSES, CLASS_COLOR, classify, rollingMean, type AaePair, type AaeRegions } from './aae'
import { dailyAae } from './AaeValues'
import { NOTEBOOK, RegionBands, provenance } from './common'
import type { DailySite } from './data'

/**
 * addis_01 Task 5 (daily variant): each daily bin's AAE through the record, coloured by its class, with
 * the notebooks' 7-bin rolling mean and the source regions behind. One panel per site.
 */
export function AaeTimeline({ daily, pair, regions }: { daily: DailySite[]; pair: AaePair; regions: AaeRegions }) {
  const panels = useMemo(() => daily.map((d) => {
    const parse = (s: string) => new Date(`${s.replace(' ', 'T')}:00`)
    const vals = dailyAae(d.daily, pair, 'absorption')
    // over every bin, as pandas rolled the notebook's series (a bin without a value still takes a place in the window)
    const roll = rollingMean(vals, 7, 3)
    const pts = d.daily.start.map((s, i) => ({ t: parse(s), v: vals[i] as number, s, m: roll[i] })).filter((p) => p.v !== null && Number.isFinite(p.v))
    const dots: SweepSeries[] = CLASSES.map((c) => ({
      id: `${d.code}-${c}`, label: c, color: CLASS_COLOR[c], width: 0, marks: true, r: 2, interval: 'none',
      points: pts.filter((p) => classify(p.v, regions) === c).map((p) => ({ x: p.t, y: p.v, tip: [`${d.name} · bin from ${p.s}`, `AAE ${pair}: ${f2(p.v)}`, c] })),
    }))
    // the mean breaks where the record does (more than ten days without a bin) rather than bridging it
    const lines: SweepSeries[] = splitAtGaps(pts, (p) => p.t, 10).map((seg, k) => ({
      id: `${d.code}-roll-${k}`, label: '7-bin rolling mean', color: d.color, width: 2, marks: false, interval: 'none',
      points: seg.map((p) => ({ x: p.t, y: p.m, tip: [`${d.name} · 7-bin rolling mean`, `to ${p.s}: ${f2(p.m)}`] })) }))
    return { d, series: [...dots, ...lines], n: pts.length }
  }), [daily, pair, regions])
  return (
    <ChartFrame id="aeth-sources-aae-timeline"
      title={`AAE through the record — each daily bin's AAE ${pair} from absorption, with a 7-bin rolling mean`}
      subtitle={`Points are coloured by their class (fossil ≤ ${regions.fossil_max}, biomass ≥ ${regions.biomass_min}); the line is the mean of each bin and the six before it, kept where at least three have a value, as addis_01_daily drew it. Gaps in the record are breaks in the instrument's data, not zeros. The y range holds the middle 98 % of each site's bins; the rest are drawn hollow at the edge.`}
      provenance={provenance([NOTEBOOK.addis01daily, NOTEBOOK.addis01], `Reads aethalometer/daily/<CODE>.json (${AAE_PAIRS[pair].daily}); addis_01 drew the same over the Jacros 60-s record, which is not in the repo.`)}>
      {!panels.length ? <Empty>No site in view has aethalometer daily bins.</Empty> : <>
        <FacetGrid items={panels} minCell={280} maxCols={2} render={(p, w) => (
          <SweepPlot key={p.d.code} width={w} height={230} title={<span style={{ color: p.d.color }}>{p.d.name} · {p.n} bins</span>}
            series={p.series} xScale="time" yLabel={`AAE ${pair}`} yClip={[0.01, 0.99]} include={[regions.fossil_max, regions.biomass_min]}
            background={({ y, iw, ih }) => <RegionBands orient="y" scale={(v) => y(v)} length={ih} depth={iw} regions={regions} />} />
        )} />
        <Key items={[...CLASSES.map((c) => ({ label: c, color: CLASS_COLOR[c] })), { label: '7-bin rolling mean (site colour)', color: INK.text, kind: 'line' as const }]} />
      </>}
    </ChartFrame>
  )
}
