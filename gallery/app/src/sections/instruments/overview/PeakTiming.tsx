import { useMemo, useRef } from 'react'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { IntervalStrip, type IntervalBlock } from '@/components/IntervalStrip'
import { Legend, useLegend } from '@/components/Legend'
import { SweepPlot, type SweepSeries } from '@/components/SweepPlot'
import { useDimensions } from '@/hooks/useGalleryData'
import { fmt, pct } from '@/lib/format'
import { INK } from '@/lib/theme'
import { seasonOnly } from '@/lib/seasons'
import { useSearchParam } from '@/lib/url'
import type { MetaFile } from '@/lib/types'
import { seasonsForSite } from '@/siteSeasons'
import { siteColor } from './data'
import { hourShares, type DayShape, type Hour } from './hourly'
import { summarize } from './stats'
import type { HourlySeries } from './useHourly'

type What = 'peak' | 'above'
type Split = 'season' | 'week'
interface Group { key: string; label: string; color: string; days: DayShape[]; hours: Hour[]; threshold: number }
/** The share of each group's days peaking at, or hours above the threshold falling in, each local hour. */
function HourLines({ groups, what, width, leg }: { groups: Group[]; what: What; width: number; leg: ReturnType<typeof useLegend> }) {
  const series: SweepSeries[] = groups.filter((g) => leg.show(g.key)).map((g) => {
    const counted = what === 'peak' ? g.days.map((d) => d.peakHour) : g.hours.filter((h) => h.v > g.threshold).map((h) => h.hour)
    const of = what === 'peak' ? `${g.days.length} days` : `${counted.length} high hours`
    return { id: g.key, label: g.label, group: g.key, color: g.color, width: 2, marks: false, gaps: true, interval: 'none' as const,
      points: hourShares(counted).map((v, h) => ({ x: h, y: v, tip: [`${hour(h)}:00 local standard time`, `${g.label}: ${pct(v)} of ${of}`] })) }
  })
  return <SweepPlot series={series} xScale="linear" xDomain={[0, 23]} xLabel="Local hour (standard time)" xFormat={(v) => hour(+v)}
    yLabel={what === 'peak' ? 'Share of days' : 'Share of high hours'} yFormat={(v) => pct(v)} include={[0]} width={width} height={278} dim={leg.dim} />
}

const hour = (h: number) => String(h).padStart(2, '0')

/**
 * When the day peaks and how far above its mean (addis_02_temporal_patterns Tasks 3 and 4 from
 * the hourly record): for each group of days, the share whose highest hour falls at each local hour,
 * or the share of the hours above the record's 95th percentile at each hour, and each day's highest
 * hour over its mean, which a 24 h filter cannot see.
 */
export function PeakTiming({ series, meta, by }: { series: HourlySeries[]; meta: MetaFile; by: 'site' | 'one' }) {
  const ref = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(ref)
  const [what, setWhat] = useSearchParam<What>('aeopk', 'peak', ['peak', 'above'])
  const [split, setSplit] = useSearchParam<Split>('aeopks', 'season', ['season', 'week'])
  const leg = useLegend()
  const groups = useMemo<Group[]>(() => {
    if (by === 'site') return series.map((hs) => ({ key: hs.site, label: hs.site, color: siteColor(meta, hs.site), days: hs.days, hours: hs.hours, threshold: hs.threshold }))
    const hs = series[0]
    if (!hs) return []
    const base = { threshold: hs.threshold }
    const parts = split === 'week'
      ? [{ key: 'Mon–Fri', label: 'Mon–Fri', color: INK.accent, f: (dow: number) => dow < 5 }, { key: 'Sat–Sun', label: 'Sat–Sun', color: INK.negative, f: (dow: number) => dow >= 5 }]
        .map((p) => ({ ...base, key: p.key, label: p.label, color: p.color, days: hs.days.filter((d) => p.f(d.dow)), hours: hs.hours.filter((h) => p.f(h.dow)) }))
      : seasonsForSite(meta, hs.site).map((se) => ({ ...base, key: seasonOnly(se.name), label: seasonOnly(se.name), color: se.color, days: hs.days.filter((d) => d.season?.name === se.name), hours: hs.hours.filter((h) => h.season?.name === se.name) }))
    return [...parts, { ...base, key: 'All days', label: 'All days', color: INK.text, days: hs.days, hours: hs.hours }].filter((g) => g.days.length || g.hours.length)
  }, [series, meta, by, split])
  const blocks = useMemo<IntervalBlock[]>(() => [{
    key: 'r', rows: groups.map((g) => {
      const s = summarize(g.days.map((d) => d.ratio))
      return { key: g.key, label: g.label, labelColor: g.color === INK.text ? undefined : g.color, note: s ? `${s.n} d` : '',
        marks: [s ? [{ mid: null, lo: s.q10, hi: s.q90, color: g.color, faint: true }, { mid: s.median, lo: s.q25, hi: s.q75, color: g.color }, { mid: s.mean, color: g.color, shape: 'hollow' as const, r: 3 }] : []],
        tip: s ? [`${g.label}: the day's highest hour over its mean`, `median ${fmt(s.median, 2)}× [IQR ${fmt(s.q25, 2)}–${fmt(s.q75, 2)}], mean ${fmt(s.mean, 2)}×`, `10–90 % ${fmt(s.q10, 2)}–${fmt(s.q90, 2)}×`, `${s.n} days with at least 18 valid hours`] : undefined }
    }),
  }], [groups])
  const thr = series.map((hs) => `${hs.site} ${fmt(hs.threshold, 1)}`).join(', ')
  const half = width > 760
  return (
    <ChartFrame id={`aeth-overview-peaks-${by}`} title={by === 'site' ? 'When the day peaks — each site\'s hourly IR BCc' : `${series[0]?.site ?? ''} · when the day peaks — and how far above its mean`}
      subtitle={`From the hourly IR BCc record (local standard time), days with at least 18 valid hours. Left: ${what === 'peak' ? "the share of days whose highest hour falls at each local hour (addis_02's peak hour, day by day)" : `the share of the hours above the record's 95th-percentile hour (${thr} µg/m³) at each local hour (addis_02's extreme events by hour)`}. Right: each day's highest hour divided by the day's mean — how much a 24 h filter average flattens the day (median and IQR, mean open, 10–90 % faint).`}
      provenance="addis_02_temporal_patterns.ipynb Tasks 3 (peak hour by season) and 4 (extreme events by hour and season), for every site with an hourly record · gallery/data/export_aeth_overview.py"
      controls={<>
        <Segmented label="Left" value={what} options={['peak', 'above'] as What[]} onChange={setWhat} optionLabel={(o) => (o === 'peak' ? "Day's peak hour" : 'Hours above P95')} />
        {by === 'one' && <Segmented label="Split" value={split} options={['season', 'week'] as Split[]} onChange={setSplit} optionLabel={(o) => (o === 'season' ? 'Seasons' : 'Weekday / weekend')} />}
      </>}>
      <div ref={ref} className="chart-wrap">
        {!groups.length ? <Empty>No hourly data in this span.</Empty> : (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, alignItems: 'flex-start' }}>
            <HourLines groups={groups} what={what} width={half ? Math.floor(width * 0.55) : width} leg={leg} />
            <div style={{ flex: 1, minWidth: 260 }}>
              <IntervalStrip blocks={blocks} panels={[{ label: "Day's highest hour / day's mean", reference: 1, format: (v) => `${fmt(v, 1)}×` }]} labelWidth={width < 600 ? 136 : 200} rowHeight={20} noteWidth={width < 600 ? 36 : 48} minPanelWidth={180} />
            </div>
          </div>
        )}
        <Legend items={groups.map((g) => ({ label: g.key, color: g.color, shape: 'line' as const }))} {...leg.props} />
      </div>
    </ChartFrame>
  )
}
