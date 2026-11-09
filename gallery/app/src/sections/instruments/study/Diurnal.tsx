import { useMemo } from 'react'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { Legend, useLegend } from '@/components/Legend'
import { SweepPlot, type SweepSeries } from '@/components/SweepPlot'
import { useFitHeight } from '@/lib/chartFit'
import { fmt, percent } from '@/lib/format'
import type { MetaFile } from '@/lib/types'
import { useSearchParam } from '@/lib/url'
import { CalendarControl, hoursLabel, seasonColour, STUDY_PROVENANCE, StudyLoading, summaryTip, useAddisStudy, useStudyCalendar } from './data'

const ID = 'aeth-addis-study-diurnal'
const TITLE = 'Jacros through the local day'
type Metric = 'mean' | 'hours' | 'coverage'
const METRICS: Metric[] = ['mean', 'hours', 'coverage']
const LABELS: Record<Metric, string> = { mean: 'Mean BC', hours: 'Valid hours', coverage: 'Study-hour coverage' }
const clock = (hour: number) => `${String(hour).padStart(2, '0')}:00`

/** Saved season × local-hour statistics with their matching observation counts. */
export function AddisStudyDiurnal(_props: { meta: MetaFile }) {
  const { data, error } = useAddisStudy()
  const [calendarId, setCalendar] = useStudyCalendar()
  const [metric, setMetric] = useSearchParam<Metric>(`${ID}.metric`, 'mean', METRICS)
  const fit = useFitHeight()
  const legend = useLegend()
  const calendar = data?.calendars[calendarId]
  const series = useMemo<SweepSeries[]>(() => (calendar?.seasons ?? []).map((season) => {
    const hours = new Map(calendar!.diurnal.filter((h) => h.season_id === season.id).map((h) => [h.hour, h]))
    return {
      id: season.id, label: season.label, group: season.label, color: seasonColour(calendar!, season.id), gaps: true, marks: true,
      points: Array.from({ length: 24 }, (_, hour) => {
        const h = hours.get(hour)
        return { x: hour, y: h ? metric === 'mean' ? h.mean_bc_ug_m3 : metric === 'hours' ? h.valid_hours : h.coverage_percent : null,
          tip: h ? [`${season.label} · ${clock(hour)}–${clock(hour + 1)} local (UTC+3)`, ...summaryTip(h)] : ['No saved hourly summary'] }
      }),
    }
  }), [calendar, metric])
  const shown = useMemo(() => series.filter((s) => !legend.hidden.has(s.label)), [series, legend.hidden])
  const items = useMemo(() => series.map((s) => ({ label: s.label, color: s.color, shape: 'line' as const })), [series])
  if (!data || !calendar) return <StudyLoading id={ID} title={TITLE} error={error} />

  return <ChartFrame id={ID} title={TITLE} provenance={STUDY_PROVENANCE}
    subtitle="Each point combines the saved IR BC values at the same local hour within one season. Every eligible hour has equal weight. The count and coverage views use exactly the same bins as the concentration view."
    evidence={<><b>{calendar.label} ({calendarId}).</b> Local standard time (UTC+3), from the partial Jacros record. Festival windows remain included.
      {' '}Hover a point for its valid-hour count, or switch to the support views.</>}
    controls={<><CalendarControl value={calendarId} onChange={setCalendar} /><Segmented label="Show" value={metric} options={METRICS} onChange={setMetric} optionLabel={(m) => LABELS[m]} /></>}>
    {!shown.some((s) => s.points.some((p) => p.y !== null)) ? <Empty>No selected seasonal series has values to draw.</Empty> : <SweepPlot series={shown}
      xScale="linear" xDomain={[0, 23]} xLabel="Local hour of day (UTC+3)" xFormat={(h) => clock(Number(h))}
      yLabel={metric === 'mean' ? 'Available-hour mean IR BC (µg/m³)' : metric === 'hours' ? 'Valid hourly observations' : 'Valid / study calendar hours (%)'}
      yDomain={metric === 'coverage' ? [0, 100] : undefined} include={[0]}
      yFormat={(v) => metric === 'coverage' ? percent(v, 0) : metric === 'hours' ? hoursLabel(v) : fmt(v, 2)}
      height={Math.max(240, fit ?? 300)} dim={legend.dim} />}
    <Legend items={items} {...legend.props} />
    <div className="legend-definition" data-export="text">
      Coverage divides each local-hour bin by all its calendar occurrences during April 2022–March 2025.
      {' '}Missing hours stay absent from concentration means; they are not filled with zero. Seasonal lines summarize observations, with no fitted event or source effect.
    </div>
  </ChartFrame>
}
