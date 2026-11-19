import { useMemo } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Select } from '@/components/ChartFrame'
import { Key } from '@/components/Key'
import { SweepPlot, type SweepSeries } from '@/components/SweepPlot'
import { useFitHeight } from '@/lib/chartFit'
import { fmt, percent } from '@/lib/format'
import { siteCodeColor } from '@/lib/siteColours'
import { INK } from '@/lib/theme'
import type { MetaFile } from '@/lib/types'
import { useSearchParam } from '@/lib/url'
import { eventHours, localAxisDate, localClockLabel, localDateLabel, STUDY_PROVENANCE, StudyLoading, useAddisStudy } from './data'

const ID = 'aeth-addis-study-events'
const TITLE = 'Festival windows in the saved record'

/** Observed 24-hour traces; absent events remain selectable and explicitly unavailable. */
export function AddisStudyEvents({ meta }: { meta: MetaFile }) {
  const { data, error } = useAddisStudy()
  const options = useMemo(() => data?.events.map((e) => e.id) ?? [], [data])
  const [picked, setPicked] = useSearchParam<string>(`${ID}.event`, 'meskel-2022', options)
  const event = data?.events.find((e) => e.id === picked) ?? data?.events[0]
  const colour = siteCodeColor(meta, data?.deployment.site_code, INK.accent)
  const fit = useFitHeight()
  const series = useMemo<SweepSeries[]>(() => event ? [{
    id: event.id, label: `${event.label} · ${event.date_local}`, color: colour, gaps: true, marks: true,
    points: eventHours(event).map(({ hour, observation }) => ({
      x: hour + 0.5, y: observation?.bc_ug_m3 ?? null,
      tip: observation ? [`${event.label} · ${localClockLabel(observation.local)} (hour start, UTC+3)`, `IR BC ${fmt(observation.bc_ug_m3, 2)} µg/m³`,
        observation.source_present ? 'Hour present in the saved record' : 'Hour outside the saved record'] : ['No saved hourly value'],
    })),
  }] : [], [event, colour])
  const key = useMemo(() => [{ label: 'Observed hourly IR BC', color: colour, kind: 'line' as const }], [colour])
  if (!data) return <StudyLoading id={ID} title={TITLE} error={error} />
  const start = event ? +localAxisDate(event.start_local) : 0

  return <ChartFrame id={ID} title={TITLE} provenance={STUDY_PROVENANCE}
    subtitle="Observed hourly IR BC over each selected festival window. Dots sit at the middle of their hour; absent values break the line. No baseline or causal event effect is estimated."
    evidence={event ? <>
      <span style={{ display: 'block' }}><b>{event.label} · {localDateLabel(event.start_local)} — {event.unavailable ? 'unavailable in the saved record' : event.incomplete ? 'incomplete window' : 'all 24 hours available'}.</b>
        {' '}{event.valid_hours} / {event.expected_hours} valid hours ({percent(event.coverage_percent, 1)}).
        {event.valid_hours > 0 && <> Available-hour mean <b>{fmt(event.mean_bc_ug_m3, 2)} µg/m³</b>; maximum {fmt(event.max_bc_ug_m3, 2)} µg/m³.</>}</span>
      <span style={{ display: 'block' }}>{localClockLabel(event.start_local)} to {localClockLabel(event.end_local_exclusive)} (end excluded), UTC+3. {event.window_rule}</span>
    </> : 'No festival windows are present in this export.'}
    controls={options.length > 0 && <Select label="Event" value={event?.id ?? ''} options={options} onChange={setPicked}
      optionLabel={(id) => { const e = data.events.find((e) => e.id === id)!; return `${e.label} · ${e.date_local}${e.unavailable ? ' · unavailable' : e.incomplete ? ` · ${e.valid_hours}/24 h` : ''}` }} />}>
    {!event || event.unavailable || event.valid_hours === 0 ? <Empty>
      {event ? `No valid hourly observations for this window. The saved Jacros record spans ${localDateLabel(data.source.first_local)} to ${localDateLabel(data.source.end_local_exclusive)}.` : 'No festival windows are available.'}
    </Empty> : <>
      <SweepPlot series={series} xScale="linear" xDomain={[0, 24]} xLabel="Local time (UTC+3)"
        xFormat={(h) => d3.utcFormat('%H:%M')(new Date(start + Number(h) * 3600000))}
        yLabel="Hourly IR BC (µg/m³)" include={[0]} yFormat={(v) => fmt(v, 1)} height={Math.max(240, fit ?? 300)} />
      <Key items={key} />
    </>}
    <div className="legend-definition" data-export="text">
      Missing hours are not zero. No new completeness threshold is imposed on event means; an incomplete window may not represent the full day.
      {data.events.some((e) => e.unavailable) && <> Unavailable windows: {data.events.filter((e) => e.unavailable).map((e) => `${e.label} ${e.date_local}`).join('; ')}.</>}
      {' '}All festival hours remain included in the seasonal and local-hour summaries on this page.
    </div>
  </ChartFrame>
}
