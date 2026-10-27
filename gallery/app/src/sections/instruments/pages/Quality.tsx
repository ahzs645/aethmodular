import { Empty, Note, Segmented } from '@/components/ChartFrame'
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
import { INSTRUMENTS, instrumentName, JACROS, useAllDays, useQualityIndex } from '../quality/data'

const TOPICS = ['coverage', 'flow', 'smoothing', 'spots'] as const
const TOPIC_LABEL: Record<string, string> = { coverage: 'Coverage', flow: 'Flow fix', smoothing: 'Smoothing', spots: 'Dual spot' }
const TOPIC_NOTE: Record<string, string> = {
  coverage: 'Which days each instrument has black carbon on, and what the legacy ±1-day matching window does to a filter day.',
  flow: 'The spot flows day by day, the flow-fix periods, and the aethalometer against HIPS and FTIR EC before and after the fix.',
  smoothing: 'Raw BCc against the instrument\'s own smoothing, and how the HIPS comparison changes as filters the smoothing moved are set aside.',
  spots: 'The dual-spot chain: attenuation on both spots, its rate, the spot ratios, roughness, the instrument K at Jacros, and raw attenuation against BCc.',
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

  const head = (
    <div className="panel weather-head">
      <div>
        <p className="subtitle" style={{ margin: 0 }}>{TOPIC_NOTE[topic]} Everything here is drawn from the legacy daily aggregates (24 h bins labelled at their start, coverage recorded but not verified; docs/matched-sample-audit-2026-09-10.md).</p>
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
        </>}
        {topic === 'flow' && <>
          <FlowSummary all={all} meta={meta} />
          <FlowTimeline code={code} days={days} meta={meta} index={q} />
          <FlowFixCompare rows={rows} meta={meta} index={q} />
          <AgreementOverTime code={code === JACROS ? 'ETAD' : code} rows={rows} meta={meta} index={q} />
        </>}
        {topic === 'smoothing' && <>
          <SmoothRaw rows={rows} meta={meta} index={q} />
          <SmoothTimeline code={code} all={all} meta={meta} index={q} />
          <div className="panel"><Note>Not drawn here: smoothing_comparison's own smoothing (5, 10, 15 and 30-minute rolling medians), its diurnal profiles and hourly IQR widths. They need the 1-minute MA350 record, which lives on Google Drive and is not in the repo.</Note></div>
        </>}
        {topic === 'spots' && <>
          <DualSpotTimeline code={code} days={days} meta={meta} index={q} />
          {all[JACROS] && <JacrosChain days={all[JACROS]} />}
          <AtnCorrelation all={all} meta={meta} />
          <div className="panel"><Note>Not drawn here: ftir_49's minute-level checks (K and K* by attenuation bin and spot age, tape-advance steps, the clean mask, the masked-day missingness benchmark, AAE order of operations, the calendar-day window) and flow_qc_analysis's minute flow fractions and QC pass rates. They need the 1-minute MA350 record, which lives on Google Drive and is not in the repo; the charts above are their daily fallbacks.</Note></div>
        </>}
      </PageToc>
    </>
  )
}
