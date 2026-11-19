import { useMemo } from 'react'
import { BarRows, type BarRow } from '@/components/BarRows'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { useFitHeight } from '@/lib/chartFit'
import { fmt, percent } from '@/lib/format'
import type { MetaFile } from '@/lib/types'
import { useSearchParam } from '@/lib/url'
import { CalendarControl, hoursLabel, seasonColour, STUDY_PROVENANCE, StudyLoading, summaryTip, useAddisStudy, useStudyCalendar } from './data'

const ID = 'aeth-addis-study-seasons'
const TITLE = 'Jacros by season'
type Metric = 'mean' | 'median' | 'coverage'
const METRICS: Metric[] = ['mean', 'median', 'coverage']
const LABELS: Record<Metric, string> = { mean: 'Mean BC', median: 'Median BC', coverage: 'Study-hour coverage' }

/** Means and medians of all eligible hourly values, never a mean of daily means. */
export function AddisStudySeasons(_props: { meta: MetaFile }) {
  const { data, error } = useAddisStudy()
  const [calendarId, setCalendar] = useStudyCalendar()
  const [metric, setMetric] = useSearchParam<Metric>(`${ID}.metric`, 'mean', METRICS)
  const fit = useFitHeight()
  const calendar = data?.calendars[calendarId]
  const rows = useMemo<BarRow[]>(() => (calendar?.seasons ?? []).map((s) => ({
    key: s.id, label: s.label, color: seasonColour(calendar!, s.id),
    value: metric === 'coverage' ? s.coverage_percent : metric === 'median' ? s.median_bc_ug_m3 : s.mean_bc_ug_m3,
    note: `n=${hoursLabel(s.valid_hours)} h`, tip: [s.label, ...summaryTip(s)],
  })), [calendar, metric])
  if (!data || !calendar) return <StudyLoading id={ID} title={TITLE} error={error} />

  return <ChartFrame id={ID} title={TITLE} provenance={STUDY_PROVENANCE}
    subtitle="Saved hourly IR BC summaries pooled within each named season. Mean BC gives each valid hour equal weight; medians and interquartile ranges describe the same hourly population. Coverage uses all calendar hours of that season in the full paper period."
    evidence={<><b>{calendar.label} ({calendarId}).</b> {calendarId === 'belg_feb' ? 'February belongs to Belg, as in the study.' : 'February belongs to Dry in this sensitivity view.'}
      {' '}Partial Jacros record; festival windows remain included. Counts beside the bars are valid hourly values.</>}
    controls={<><CalendarControl value={calendarId} onChange={setCalendar} /><Segmented label="Show" value={metric} options={METRICS} onChange={setMetric} optionLabel={(m) => LABELS[m]} /></>}>
    {!rows.some((r) => r.value !== null) ? <Empty>No valid hourly values for these seasonal summaries.</Empty> : <BarRows rows={rows}
      rowHeight={fit === null ? 44 : Math.max(32, (fit - 46) / Math.max(1, rows.length))} barHeight={20}
      domain={metric === 'coverage' ? [0, 100] : undefined} maxLabelWidth={260}
      format={(v) => metric === 'coverage' ? percent(v, 1) : fmt(v, 2)}
      axis={{ label: metric === 'coverage' ? 'Valid hours / study calendar hours (%)' : `${metric === 'median' ? 'Median' : 'Available-hour mean'} IR BC (µg/m³)`, ticks: 5 }} />}
    <div className="legend-definition" data-export="text">
      {calendar.seasons.map((s) => `${s.label}: ${hoursLabel(s.valid_hours)} / ${hoursLabel(s.expected_hours)} calendar hours (${percent(s.coverage_percent, 1)})`).join(' · ')}.
      {' '}These are seasonal summaries of the saved local record; the paper’s seasonal exclusions and complete record are not reproduced.
    </div>
  </ChartFrame>
}
