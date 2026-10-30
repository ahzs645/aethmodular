import { Empty, Segmented } from '@/components/ChartFrame'
import { PageToc } from '@/components/PageToc'
import { useSearchParam } from '@/lib/url'
import type { SectionPageProps } from '../../types'
import { CoverageStrip } from '../quality/CoverageStrip'
import { FlowSummary, FlowTimeline } from '../quality/FlowCharts'
import { FlowFixCompare } from '../quality/FlowFixCompare'
import { AgreementOverTime } from '../quality/AgreementOverTime'
import { SmoothRaw, SmoothTimeline } from '../quality/SmoothRaw'
import { AtnCorrelation } from '../quality/AtnCorrelation'
import { DualSpotTimeline, JacrosChain, WindowEffect } from '../quality/DualSpot'
import { INSTRUMENTS, instrumentName, JACROS, useAllDays, useQualityIndex, type Day } from '../quality/data'
import { minuteCode, useMinuteFlow, useMinuteIndex, useMinuteSites } from '../quality/minute'
import { FlowRatioSite, FlowRatioSites, MinuteFlowClasses, MinuteFlowDiffs, MinuteFlowFractions, MinuteFlowHourly, MinuteRDistribution, useFlowRecords } from '../quality/FlowQcCharts'
import { MinuteAaeOrder, MinuteClean, MinuteFlowQC, MinuteKstar, MinuteMissing, MinuteSmoothing, MinuteTape, MinuteWindows } from '../quality/MinuteCharts'

/** a stable empty record set while the daily bins load */
const NO_DAYS: Record<string, Day[]> = {}
const TOPICS = ['coverage', 'flow', 'smoothing', 'spots'] as const
const TOPIC_LABEL: Record<string, string> = { coverage: 'Coverage', flow: 'Flow fix', smoothing: 'Smoothing', spots: 'Dual spot' }
const TOPIC_NOTE: Record<string, string> = {
  coverage: 'Which days each instrument has black carbon on, and what the legacy ±1-day matching window does to a filter day.',
  flow: 'The spot flows day by day, the flow-fix periods, the flow ratio at every instrument and, on the two minute records, flow_qc_analysis\'s hourly ratio, flow fractions, flow differences and ratio distribution, and the aethalometer against HIPS and FTIR EC before and after the fix.',
  smoothing: 'Raw BCc against the smoothed BCc of the pickles (the cleaning pipeline\'s DEMA) and the notebook\'s rolling medians, and how the HIPS comparison changes as filters the smoothing moved are set aside.',
  spots: 'The dual-spot chain: attenuation on both spots, its rate, the spot ratios, roughness, the instrument K at Addis, and raw attenuation against BCc.',
}

/**
 * Sub-page 'quality' of the Aethalometer tab: the QC notebooks (FlowFix_BeforeAfter_Analysis, flow_fix_explorer,
 * notebooks/qc/flow_qc_analysis, HIPS_Aeth_SmoothRaw_Analysis, smoothing_comparison, Raw_ATN_Correlation_Analysis,
 * notebooks/plotting_gaps_scenarios, research/ftir_ec_phase3/ftir_49) on the daily bins the repo holds
 * (gallery/data/export_aethalometer.py, gallery/data/export_aeth_quality.py), one topic at a time. Charts about
 * one instrument follow the Instrument picker; the per-filter ones use the tab's rows, so the Filters panel's
 * dates, seasons and exclusions apply.
 */
export function Quality({ rows, meta, picked }: SectionPageProps) {
  const index = useQualityIndex()
  const all = useAllDays()
  const [topic, setTopic] = useSearchParam<string>('aq.topic', 'coverage', [...TOPICS])
  const available = INSTRUMENTS.filter((c) => !all || all[c])
  const firstPicked = available.find((c) => picked.has(c)) ?? 'USPA'
  const [code, setCode] = useSearchParam<string>('aq.site', firstPicked, [...available])
  const days = all?.[code] ?? []
  const minuteIndex = useMinuteIndex().data
  const minute = useMinuteSites(minuteIndex)
  const mc = minuteCode(code)
  const minuteNames = Object.fromEntries(Object.entries(minuteIndex?.sites ?? {}).map(([k, v]) => [k, k === JACROS ? instrumentName(k, meta) : v.name]))
  const mName = minuteNames[mc] ?? instrumentName(code, meta)
  const mSite = minute[mc] ?? null
  const mFlow = useMinuteFlow(topic === 'flow' ? mc : null, minuteIndex ?? null).data
  const uspaFlow = useMinuteFlow(topic === 'flow' ? 'USPA' : null, minuteIndex ?? null).data
  const flowRecords = useFlowRecords(all ?? NO_DAYS, meta, uspaFlow)

  const head = (
    <div className="panel weather-head">
      <div>
        <p className="subtitle" style={{ margin: 0 }}>{TOPIC_NOTE[topic]} The daily charts use the legacy daily aggregates (24 h bins labelled at their start, coverage recorded but not verified; docs/matched-sample-audit-2026-09-10.md); the minute-level charts use the raw MA350 records at Addis and Pasadena (JPL), the only two in the Drive mirror. Addis is one instrument (MA350-0238): its Jacros file is the full record and holds every day of the Addis file, so the page reads Addis from it.</p>
      </div>
      <div className="weather-head-controls">
        <Segmented label="Topic" value={topic} options={TOPICS} onChange={setTopic} optionLabel={(t) => TOPIC_LABEL[t]} />
        {topic !== 'coverage' && <Segmented label="Instrument" value={code} options={available} onChange={setCode} optionLabel={(c) => instrumentName(c, meta)} />}
      </div>
    </div>
  )
  if (index.error) return <>{head}<div className="panel"><Empty>The quality export is missing ({index.error}). Run uv run python gallery/data/export_aeth_quality.py.</Empty></div></>
  if (!index.data || !all) return <>{head}<div className="panel"><Empty>Loading the daily bins and their QC columns…</Empty></div></>
  const q = index.data

  return (
    <>
      {head}
      <PageToc>
        {topic === 'coverage' && <>
          <CoverageStrip all={all} meta={meta} index={q} />
          <WindowEffect all={all} rows={rows} meta={meta} />
          <MinuteWindows sites={minute} names={minuteNames} meta={meta} rows={rows} />
          <MinuteMissing sites={minute} names={minuteNames} />
        </>}
        {topic === 'flow' && <>
          <FlowSummary all={all} meta={meta} />
          <FlowTimeline code={code} days={days} meta={meta} index={q} />
          <FlowRatioSites records={flowRecords} />
          <FlowRatioSite code={code} records={flowRecords.instrument} meta={meta} />
          <MinuteFlowQC site={mSite} name={mName} />
          <MinuteFlowHourly flow={mFlow} name={mName} />
          <MinuteFlowFractions flow={mFlow} name={mName} />
          <MinuteFlowDiffs site={mSite} flow={mFlow} name={mName} />
          <MinuteRDistribution site={mSite} flow={mFlow} name={mName} />
          <MinuteFlowClasses site={mSite} name={mName} />
          <FlowFixCompare rows={rows} meta={meta} index={q} />
          <AgreementOverTime code={code} rows={rows} meta={meta} index={q} />
        </>}
        {topic === 'smoothing' && <>
          <SmoothRaw rows={rows} meta={meta} index={q} />
          <SmoothTimeline code={code} all={all} meta={meta} index={q} />
          <MinuteSmoothing site={mSite} name={mName} />
        </>}
        {topic === 'spots' && <>
          <DualSpotTimeline code={code} days={days} meta={meta} index={q} />
          {all.ETAD && <JacrosChain days={all.ETAD} />}
          <AtnCorrelation all={all} meta={meta} />
          <MinuteKstar site={mSite} name={mName} />
          <MinuteTape site={mSite} name={mName} />
          <MinuteClean site={mSite} name={mName} />
          <MinuteAaeOrder site={mSite} name={mName} />
        </>}
      </PageToc>
    </>
  )
}
