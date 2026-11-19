import { useMemo } from 'react'
import * as d3 from 'd3'
import { ChartFrame, Empty, Segmented } from '@/components/ChartFrame'
import { Key } from '@/components/Key'
import { RefBand } from '@/components/RefLines'
import { StackedPanels } from '@/components/StackedPanels'
import { robustDomain } from '@/lib/axes'
import { fmt, percent } from '@/lib/format'
import { siteCodeColor } from '@/lib/siteColours'
import { lineRuns, nearestIndex } from '@/lib/stacked'
import { INK } from '@/lib/theme'
import type { MetaFile } from '@/lib/types'
import { useSearchParam } from '@/lib/url'
import { hoursLabel, localAxisDate, localRangeLabel, STUDY_PROVENANCE, StudyLoading, summaryTip, useAddisStudy } from './data'

const ID = 'aeth-addis-study-timeline'
const TITLE = 'Jacros record and its gaps'
const BINS = ['daily', 'monthly'] as const

/** A saved-hourly comparison, independent of the gallery's filter-sample selection. */
export function AddisStudyTimeline({ meta }: { meta: MetaFile }) {
  const { data, error } = useAddisStudy()
  const [bin, setBin] = useSearchParam<(typeof BINS)[number]>(`${ID}.bin`, 'daily', BINS)
  const points = useMemo(() => (data?.[bin] ?? []).map((s) => {
    const start = localAxisDate(s.start_local), end = localAxisDate(s.end_local_exclusive)
    return { ...s, start, end, middle: new Date((+start + +end) / 2) }
  }), [data, bin])
  const runs = useMemo(() => lineRuns(points, (s) => s.middle, (s) => s.mean_bc_ug_m3, bin === 'daily' ? 1.1 : 32), [points, bin])
  const domain = useMemo<[Date, Date]>(() => data
    ? [localAxisDate(data.period.start_local), localAxisDate(data.period.end_local_exclusive)] : [new Date(0), new Date(1)], [data])
  const yDomain = useMemo(() => robustDomain(points.map((p) => p.mean_bc_ug_m3).filter((v): v is number => v !== null), 0, 1, { include: [0] }).domain, [points])
  const colour = siteCodeColor(meta, data?.deployment.site_code, INK.accent)
  const legend = useMemo(() => [
    { label: 'Available-hour mean IR BC', color: colour, kind: 'line' as const },
    { label: 'Outside the saved record', color: INK.accent, kind: 'faded' as const },
  ], [colour])
  if (!data) return <StudyLoading id={ID} title={TITLE} error={error} />
  const cov = data.coverage
  const snapshotStart = localAxisDate(data.source.first_local), snapshotEnd = localAxisDate(data.source.end_local_exclusive)

  return <ChartFrame id={ID} title={TITLE} provenance={STUDY_PROVENANCE}
    subtitle="Daily or monthly available-hour means from the saved IR record, with the full paper period retained on the horizontal axis. Missing values break the line. Coverage is finite hourly values divided by all calendar hours in each bin, including hours beyond the saved record."
    evidence={<>
      <span style={{ display: 'block' }}><b>Partial Jacros record · {data.deployment.serial}.</b> {localRangeLabel(data.source.first_local, data.source.end_local_exclusive)} · local standard time (UTC+3).</span>
      <span style={{ display: 'block' }}>{hoursLabel(cov.valid_hours)} valid hours: <b>{percent(cov.coverage_percent, 1)}</b> of {hoursLabel(cov.expected_hours)} calendar hours in the paper period;
        {' '}<b>{percent(cov.snapshot_coverage_percent, 1)}</b> of {hoursLabel(cov.snapshot_hours)} saved slots. Available-hour mean <b>{fmt(data.summary.mean_bc_ug_m3, 2)} µg/m³</b>.</span>
    </>}
    controls={<Segmented label="Bins" value={bin} options={BINS} onChange={setBin} optionLabel={(v) => v === 'daily' ? 'Daily' : 'Monthly'} />}>
    {!points.length ? <Empty>No saved timeline summaries are available.</Empty> : <StackedPanels
      x={{ type: 'time', domain, utc: true }}
      margin={{ top: 12, bottom: 46, left: 70 }} gap={22}
      ariaLabel={`${bin === 'daily' ? 'Daily' : 'Monthly'} Jacros IR BC and valid-hour coverage over the full study period`}
      xAxis={{ label: 'Addis local date (UTC+3)', format: d3.utcFormat('%b %Y'), tickCount: 7 }}
      panels={[
        {
          key: 'bc', height: 220, share: 3,
          y: { domain: yDomain, nice: true, label: 'IR BC (µg/m³)', labelX: -50, tickCount: 5 },
          background: ({ x, width, height }) => <>
            <RefBand orient="v" at={[0, x(snapshotStart)]} size={width} length={height} />
            <RefBand orient="v" at={[x(snapshotEnd), width]} size={width} length={height} />
          </>,
          children: ({ x, y, clip }) => <g clipPath={clip}>
            {runs.map((run, i) => run.length === 1
              ? <circle key={i} cx={x(run[0].t)} cy={y(run[0].v)} r={2.4} fill={colour} />
              : <path key={i} d={d3.line<(typeof run)[number]>().x((p) => x(p.t)).y((p) => y(p.v))(run) ?? ''} fill="none" stroke={colour} strokeWidth={bin === 'daily' ? 1.1 : 1.9} />)}
          </g>,
        },
        {
          key: 'coverage', height: 72, share: 1,
          y: { domain: [0, 100], label: 'Valid hours (%)', labelX: -50, tickCount: 3 },
          background: ({ x, width, height }) => <>
            <RefBand orient="v" at={[0, x(snapshotStart)]} size={width} length={height} />
            <RefBand orient="v" at={[x(snapshotEnd), width]} size={width} length={height} />
          </>,
          children: ({ x, y, height, clip }) => <g clipPath={clip}>
            {points.filter((p) => p.coverage_percent !== null && p.coverage_percent > 0).map((p) => <rect key={p.start_local}
              x={x(p.start)} width={Math.max(0.4, x(p.end) - x(p.start) - 0.3)} y={y(p.coverage_percent!)} height={height - y(p.coverage_percent!)} fill={colour} fillOpacity={0.65} />)}
          </g>,
        },
      ]}
      readout={(px, x) => {
        const point = points[nearestIndex(points, (p) => +p.middle, +x.invert(px))]
        return point ? { at: x(point.middle), tip: [localRangeLabel(point.start_local, point.end_local_exclusive), ...summaryTip(point)] } : null
      }} />}
    <Key items={legend} />
    <div className="legend-definition" data-export="text">
      Each retained hour had at least {data.source.min_valid_ir_minutes} valid IR minute records in the upstream export. No additional upper-value screen or daily completeness gate is applied here.
      {' '}The line is an available-hour mean; the lower panel shows how much of each bin supports it.
    </div>
  </ChartFrame>
}
