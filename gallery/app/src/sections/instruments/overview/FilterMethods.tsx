import { useMemo, useRef } from 'react'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { IntervalStrip, type IntervalBlock, type IntervalMark } from '@/components/IntervalStrip'
import { Key } from '@/components/Key'
import { useDimensions } from '@/hooks/useGalleryData'
import { fmt, signed } from '@/lib/format'
import { INK } from '@/lib/theme'
import { seasonOnly } from '@/lib/seasons'
import { useSearchParam } from '@/lib/url'
import type { FilterRow, MetaFile } from '@/lib/types'
import { seasonsForSite } from '@/siteSeasons'
import { FILTER_METHODS, type Series } from './data'
import { MONTH_STAT_LABEL, MonthPanel, type MonthStat } from './MonthPanel'
import { compareWeekend, summarize, type Summary } from './stats'

type View = 'month' | 'season' | 'week'
const VIEW_LABEL: Record<View, string> = { month: 'By month', season: 'By season', week: 'Weekday / weekend' }
const val = (r: FilterRow, field: string) => (typeof r[field] === 'number' && Number.isFinite(r[field]) ? (r[field] as number) : null)
/** 0 = Monday, of the filter's sample date */
const dow = (date: string) => (new Date(`${date}T00:00:00Z`).getUTCDay() + 6) % 7
const marks = (s: Summary | null, color: string): IntervalMark[] => (s ? [
  { mid: s.median, lo: s.q25, hi: s.q75, color }, { mid: s.mean, color, shape: 'hollow', r: 3 }] : [])
const tipOf = (title: string, s: Summary) => [title, `n = ${s.n} filters`, `mean ± SD ${fmt(s.mean, 2)} ± ${fmt(s.sd, 2)} µg/m³`, `median ${fmt(s.median, 2)} [IQR ${fmt(s.q25, 2)}–${fmt(s.q75, 2)}]`]

/**
 * The same filters measured three ways, by month, season and weekday (addis_02.5_temporal_patterns_filter
 * Tasks 1–3): the aethalometer's IR BCc over each filter's window, FTIR EC and HIPS Fabs / MAC.
 */
export function FilterMethods({ s, meta }: { s: Series; meta: MetaFile }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const { width } = useDimensions(wrapRef)
  const [view, setView] = useSearchParam<View>('aeof', 'month', ['month', 'season', 'week'])
  const [stat, setStat] = useSearchParam<MonthStat>('aeom', 'mean', ['mean', 'median'])
  const rows = s.rows
  const counts = FILTER_METHODS.map((m) => rows.filter((r) => val(r, m.field) !== null).length)
  const all3 = rows.filter((r) => FILTER_METHODS.every((m) => val(r, m.field) !== null)).length
  const seasons = seasonsForSite(meta, s.inst.site)

  const monthSeries = useMemo(() => FILTER_METHODS.map((m) => ({ key: m.key, label: m.label, color: m.color,
    months: Array.from({ length: 12 }, (_, i) => summarize(rows.filter((r) => r.month === i + 1).map((r) => val(r, m.field)))) })), [rows])
  const seasonBlocks = useMemo<IntervalBlock[]>(() => [...seasons.map((se) => ({ key: se.name, title: seasonOnly(se.name), color: se.color as string | undefined, rows: rows.filter((r) => r.season === se.name) })),
    { key: 'all', title: 'All filters', color: undefined, rows }].filter((g) => g.rows.length).map((g) => ({
    key: g.key, title: g.title, color: g.color,
    rows: FILTER_METHODS.map((m) => {
      const st = summarize(g.rows.map((r) => val(r, m.field)))
      return { key: m.key, label: m.label, labelColor: m.color, marks: [marks(st, m.color)], note: st ? `n=${st.n}` : '', tip: st ? tipOf(`${g.title} · ${m.label}`, st) : undefined }
    }),
  })), [rows, seasons])
  const weekBlocks = useMemo<IntervalBlock[]>(() => FILTER_METHODS.map((m) => {
    const wd = rows.filter((r) => dow(r.date) < 5).map((r) => val(r, m.field)).filter((v): v is number => v !== null)
    const we = rows.filter((r) => dow(r.date) >= 5).map((r) => val(r, m.field)).filter((v): v is number => v !== null)
    const w = compareWeekend(wd, we)
    const a = summarize(wd), b = summarize(we)
    return { key: m.key, title: m.label, color: m.color, rows: [
      { key: 'wd', label: 'Mon–Fri', marks: [marks(a, INK.accent), []], note: a ? `n=${a.n}` : '', tip: a ? tipOf(`${m.label} · Mon–Fri`, a) : undefined },
      { key: 'we', label: 'Sat–Sun', marks: [marks(b, INK.negative), []], note: b ? `n=${b.n}` : '', tip: b ? tipOf(`${m.label} · Sat–Sun`, b) : undefined },
      { key: 'fx', label: 'Weekend vs weekdays', strong: true, marks: [[], w ? [{ mid: w.change, lo: w.lo, hi: w.hi, color: m.color, shape: 'diamond' as const, r: 5 }] : []],
        note: w ? (w.p < 0.001 ? 'p < 0.001' : `p = ${fmt(w.p, 2)}`) : '', tip: w ? [`${m.label}: weekend ${signed(w.change, 1)} % of the weekday mean`, `95 % ${signed(w.lo, 1)} to ${signed(w.hi, 1)}, Welch p = ${fmt(w.p, 3)}`] : undefined },
    ] }
  }), [rows])

  return (
    <ChartFrame id="aeth-overview-filter-methods" title={`${s.inst.site} · the filters three ways — aethalometer, FTIR EC and HIPS by month, season and weekday`}
      subtitle={`The site's filters in view (${rows.length}; ${counts.map((n, i) => `${FILTER_METHODS[i].label} ${n}`).join(', ')}; all three ${all3}). The aethalometer value is the mean of the daily bins over each filter's own sampling window, weighted by overlap and kept where they cover at least 12 h (the notebook matched one 9 AM day within ±1 day of the sample date); HIPS is Fabs / MAC with MAC = ${meta.mac_value} m²/g. ${view === 'month' ? `Per calendar month of the sample date: ${MONTH_STAT_LABEL[stat].toLowerCase()}.` : view === 'season' ? 'Per season: median and interquartile range, mean open.' : "Weekdays against weekends by sample date, and the weekend's mean as a % of the weekday mean (95 % interval from the Welch standard error)."} Each method keeps every filter it has, as the notebook did, so the sets differ a little.`}
      provenance="addis_02.5_temporal_patterns_filter.ipynb Tasks 1–3 (monthly, weekly and seasonal patterns for the three methods), for every site · per-filter values: gallery/data/export_aethalometer.py"
      controls={<>
        <Segmented label="View" value={view} options={['month', 'season', 'week'] as View[]} onChange={setView} optionLabel={(o) => VIEW_LABEL[o]} />
        {view === 'month' && <Segmented label="Statistic" value={stat} options={['mean', 'median'] as MonthStat[]} onChange={setStat} optionLabel={(o) => MONTH_STAT_LABEL[o]} />}
      </>}>
      <div ref={wrapRef} className="chart-wrap">
        {!rows.length ? <Empty>No filters at this site in view.</Empty>
          : view === 'month' ? <MonthPanel series={monthSeries} seasons={seasons} stat={stat} width={Math.min(width, 900)} height={280} unit="µg/m³" />
          : view === 'season' ? <IntervalStrip blocks={seasonBlocks} panels={[{ label: 'µg/m³' }]} labelWidth={170} rowHeight={17} noteWidth={56} />
          : <IntervalStrip blocks={weekBlocks} panels={[{ label: 'µg/m³', heading: 'Filters' }, { label: 'Weekend − weekdays (% of weekday mean)', reference: 0, heading: 'Weekend effect', format: (v) => `${v > 0 ? '+' : ''}${Math.round(v)}` }]}
            labelWidth={150} rowHeight={17} noteWidth={74} minPanelWidth={220} />}
        <Key items={FILTER_METHODS.map((m) => ({ label: m.label, color: m.color, kind: view === 'month' ? 'line' as const : 'dot' as const }))} />
      </div>
    </ChartFrame>
  )
}
