import { useMemo } from 'react'
import { ChartFrame, Empty, Select, Segmented } from '@/components/ChartFrame'
import { LoadState } from '@/components/ExportState'
import { ScatterPlot, type ScatterPoint } from '@/components/ScatterPlot'
import { BarRows, type BarRow } from '@/components/BarRows'
import { Legend, useLegend } from '@/components/Legend'
import { useSearchParam } from '@/lib/url'
import { useFitHeight } from '@/lib/chartFit'
import { ExcludedNote, FixedSetNote } from '@/lib/markedRows'
import { fmt, signed } from '@/lib/format'
import { CONVENTION_LABEL, seasonOfDate } from '@/lib/seasons'
import { finite, labCohortRows, LAB_RESPONSE, LAB_SOURCE, type LabFilter } from '../../common/baselineLab'
import { ML_CALENDARS, type MLCalendar } from '../../common/seasonalML'
import { labSiteColor, labSiteName, LabSourceSelect, SOURCE_CHOICES, sourceIncludes, useLabExport, useLabFilters,
  useLabSelection, type LabChartProps, type LabSource } from './data'

const VIEWS = ['timeline', 'lot', 'season', 'matched cohorts'] as const

/** Ethiopia is compared within dated populations; absent lot metadata stays explicitly unknown. */
export function BaselineEthiopia({ rows, meta }: LabChartProps) {
  const { data, error } = useLabExport()
  const scoped = useLabFilters(data, rows)
  const [view, setView] = useSearchParam<typeof VIEWS[number]>('blab.ethiopia.view', 'matched cohorts', VIEWS)
  const [source, setSource] = useSearchParam<LabSource>('blab.source', 'direct', SOURCE_CHOICES)
  const [wanted, setComparison] = useSearchParam<string>('blab.ethiopia.comparison', '')
  const [metric, setMetric] = useSearchParam<string>('blab.ethiopia.metric', 'signed', ['signed', 'absolute'])
  const [calendar, setCalendar] = useSearchParam<MLCalendar>('blab.ethiopia.calendar', 'belg_feb', ML_CALENDARS)
  const calendarSeasons = meta.season_conventions[calendar] ?? []
  const [, selectFilter] = useLabSelection()
  const height = useFitHeight() ?? 345
  const lg = useLegend()
  const selected = useMemo(() => {
    const use = (r: LabFilter) => ['ETAD', 'ETBI'].includes(r.code) && r.type === 'PM2.5'
      && (r.spectral_pair_eligible || r.is_excluded) && sourceIncludes(r, source) && finite(r.optical_disagreement_ugm3)
      && (view !== 'timeline' || !!r.date)
    return { counted: scoped.counted.filter(use), marked: scoped.marked.filter(use) }
  }, [scoped, source, view])
  const seasonOf = (r: LabFilter) => seasonOfDate(r.date, r.date ? Number(r.date.slice(5, 7)) : null, calendarSeasons)?.name ?? null
  const categoryOf = (r: LabFilter) => view === 'lot' ? `${r.code} · lot ${r.lot_group || 'unknown'}`
    : `${r.code} · ${seasonOf(r) ?? 'season unknown'}`
  const categories = [...new Set(selected.counted.map(categoryOf))].sort()
  const points = useMemo<ScatterPoint[]>(() => data ? [...selected.counted, ...selected.marked].filter(r => lg.show(labSiteName(data, r.code))).map((r, i) => ({
    id: r.id, key: r.id, x: view === 'timeline' ? Date.parse(`${r.date}T12:00:00Z`) : categories.indexOf(categoryOf(r)) + .5,
    y: r.optical_disagreement_ugm3!, dx: view === 'timeline' ? 0 : ((i % 11) - 5) * 2.5,
    color: labSiteColor(data, r.code), hollow: r.optical_source !== 'direct_hips_fabs', dim: lg.dim(labSiteName(data, r.code)),
    tip: [r.id, `${labSiteName(data, r.code)} · ${r.date ?? 'date unavailable'}`, `Lot: ${r.lot_group || 'unknown'}`,
      `${LAB_RESPONSE}: ${signed(r.optical_disagreement_ugm3, 3)}`, `Optical source: ${r.optical_source.replace(/_/g, ' ')}`,
      `PM2.5: ${fmt(r.pm25_ugm3, 2)} µg/m³ · volume: ${fmt(r.volume_m3, 2)} m³`,
      ...(seasonOf(r) ? [`${calendar}: ${seasonOf(r)}`] : [])],
  })) : [], [data, selected, view, categories.join('|'), calendar, calendarSeasons, lg.show, lg.dim]) // eslint-disable-line react-hooks/exhaustive-deps
  const comparisons = [...new Set(data?.cohorts.map(r => r.comparison) ?? [])]
  const comparison = comparisons.includes(wanted) ? wanted : comparisons.find(c => /match/i.test(c)) ?? comparisons[0] ?? wanted
  const { available, unavailable: empty } = labCohortRows(data?.cohorts ?? [], comparison, metric === 'absolute')
  const bars: BarRow[] = !data ? [] : available.map((r, i) => ({
    key: `${r.code}-${r.group}-${i}`, label: `${r.group} · n=${r.n}`, value: metric === 'signed' ? r.median_disagreement : r.median_abs_disagreement,
    // Put counts in the label gutter; negative-bar value tails can overlap it.
    valueLabel: null, color: labSiteColor(data, r.code), tip: [r.group, comparison, labSiteName(data, r.code), `${r.n} physical filters`,
      `Median FTIR − optical: ${signed(r.median_disagreement, 3)} µg/m³`, `Median |difference|: ${fmt(r.median_abs_disagreement, 3)} µg/m³`,
      ...(r.date_start ? [`${r.date_start} to ${r.date_end ?? '—'}`] : []), ...(r.note ? [r.note] : [])],
  }))
  return <ChartFrame id="baseline-lab-ethiopia" title="Ethiopia: dates, lots and matched filters"
    source={`${LAB_SOURCE}/Ethiopia.tsx › BaselineEthiopia`}
    subtitle="Current products at Addis and Bishoftu. Matching on optical loading, PM2.5, volume and dates does not supply independent EC truth."
    tip={<><p>Timeline, lot and season views follow Samples and its shared exclusions. The live season view has its own named calendar: February belongs to Belg by default, with the historical dry_feb convention available. Saved matched-cohort summaries retain their exported membership and named calendars. Hover for counts, dates and matching rules.</p><p>Bishoftu's packed spectra cover 20 October 2025–19 February 2026. Its apparent Belg observations under belg_feb are February only and use the public HIPS extension; they cannot establish Kiremt transfer. Later Addis lot metadata are missing; a missing lot is never inferred from collection date.</p><p>A contemporaneous same-known-lot match can be empty even when dated optical pairs exist. A descriptive matched comparison does not identify whether FTIR, HIPS or a MAC assumption causes the difference.</p></>}
    controls={<><Select label="View" value={view} options={[...VIEWS]} onChange={v => setView(v as typeof VIEWS[number])} />
      {view === 'matched cohorts' ? <><Select label="Comparison" value={comparison} options={comparisons} onChange={setComparison} optionLabel={v => v.replace(/_/g, ' ')} />
        <Segmented label="Difference" value={metric} options={['signed', 'absolute']} onChange={setMetric} /></>
        : <><LabSourceSelect value={source} onChange={setSource} />
          {view === 'season' && <Select label="Calendar" value={calendar} options={[...ML_CALENDARS]} onChange={v => setCalendar(v as MLCalendar)}
            optionLabel={v => CONVENTION_LABEL[v]?.split(' · ')[1] ?? v} />}</>}</>}>
    {!data ? <LoadState what="Ethiopian comparison" error={error} /> : view === 'matched cohorts' ? <>
      {!bars.length ? <Empty>{empty[0]?.note ?? 'No comparable filters meet this saved matching rule.'}</Empty> : <div className="chart-wrap">
        <BarRows rows={bars} axis={{ label: metric === 'signed' ? 'Median FTIR − optical BC (µg/m³)' : 'Median |FTIR − optical BC| (µg/m³)' }}
          format={v => fmt(v, 3)} rowHeight={Math.max(23, Math.min(42, (height - 45) / Math.max(1, bars.length)))} />
      </div>}
      {bars.length > 0 && empty.length > 0 && <div className="chart-note">{empty.map(r => `${r.group}: n=${r.n}, no estimate${r.note ? ` — ${r.note}` : ''}`).join(' · ')}</div>}
      <FixedSetNote what="the saved date/loading-matched physical-filter cohorts" />
    </> : !points.length ? <Empty>No Ethiopian pairs in the current Samples and optical-source selection.</Empty> : <>
      <div className="chart-note">{selected.counted.length} physical pairs · {view === 'season' ? `${calendar} month bins` : 'missing lots retained as unknown'}</div>
      <div className="chart-wrap">
        <ScatterPlot points={points} xScale={view === 'timeline' ? 'time' : 'linear'}
          xBands={view === 'timeline' ? undefined : categories.map(c => ({ key: c, label: c }))}
          xLabel={view === 'timeline' ? 'Collection date' : view === 'lot' ? 'Site and recorded filter lot' : 'Site and collection season'}
          yLabel={LAB_RESPONSE} height={height} refY={[0]} note={false}
          onPick={p => { if (p.id) selectFilter(p.id) }} />
        <Legend items={['ETAD', 'ETBI'].map(code => ({ label: labSiteName(data, code), color: labSiteColor(data, code) }))} {...lg.props} />
      </div>
      <ExcludedNote count={selected.marked.length} drawn />
    </>}
  </ChartFrame>
}
