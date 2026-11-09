/** Register through the shared Dashboard, only when Instruments is compiled. */
import type { ComponentType } from 'react'
import { registerDashboardChart } from '@/components/dashboard/registry'
import type { MetaFile } from '@/lib/types'
import { AddisStudyTimeline } from './Timeline'
import { AddisStudySeasons } from './Seasonal'
import { AddisStudyDiurnal } from './Diurnal'
import { AddisStudyEvents } from './Events'
import { AddisStudyBenchmarks, AddisStudyReadiness } from './Context'
import { AddisStudyHistoricalFits, AddisStudyLoadingSensitivity, AddisStudyOpticalSensitivity, AddisStudySourceMaterials } from './Methods'
import { STUDY_PATH } from './data'

const charts: { kind: string; label: string; blurb: string; Chart: ComponentType<{ meta: MetaFile }>; size: [number, number] }[] = [
  { kind: 'aeth-addis-study-timeline', label: 'Addis study: record and coverage', blurb: 'saved Jacros IR BC and retained-hour coverage over the full paper calendar', Chart: AddisStudyTimeline, size: [12, 6] },
  { kind: 'aeth-addis-study-seasons', label: 'Addis study: seasonal BC', blurb: 'available-hour mean, median and coverage with explicit February calendar sensitivity', Chart: AddisStudySeasons, size: [6, 5] },
  { kind: 'aeth-addis-study-diurnal', label: 'Addis study: local-hour patterns', blurb: 'seasonal IR BC by local hour, with hourly support and descriptive spread', Chart: AddisStudyDiurnal, size: [6, 6] },
  { kind: 'aeth-addis-study-events', label: 'Addis study: festival traces', blurb: 'Meskel and Hidar display windows with missing hours and incomplete records visible', Chart: AddisStudyEvents, size: [8, 6] },
  { kind: 'aeth-addis-study-benchmarks', label: 'Addis study: published context', blurb: 'reported Central and Jacros statistics kept separate from the saved partial record', Chart: AddisStudyBenchmarks, size: [12, 5] },
  { kind: 'aeth-addis-study-readiness', label: 'Addis study: implementation status', blurb: 'what is connected and which observations and methods are still needed', Chart: AddisStudyReadiness, size: [12, 6] },
  { kind: 'aeth-addis-study-materials', label: 'Addis study: supplied materials', blurb: 'what the EC/BC presentation and JPL archive contribute, with provenance and remaining inputs', Chart: () => <AddisStudySourceMaterials />, size: [12, 7] },
  { kind: 'aeth-addis-study-optical-sensitivity', label: 'Addis study: archived source sensitivity', blurb: 'select saved Central/Jacros alpha-sensitivity summaries from the archive; no new source fractions', Chart: AddisStudyOpticalSensitivity, size: [12, 6] },
  { kind: 'aeth-addis-study-loading-sensitivity', label: 'Addis study: loading sensitivity', blurb: 'analytical response to three historical constant-k candidates; not applied to observations', Chart: AddisStudyLoadingSensitivity, size: [8, 6] },
  { kind: 'aeth-addis-study-historical-fits', label: 'Addis study: historical EC/BC fits', blurb: 'presentation fit summaries against FTIR-derived EC with explicit evidence limits', Chart: AddisStudyHistoricalFits, size: [12, 5] },
]

for (const { kind, label, blurb, Chart, size } of charts) registerDashboardChart({
  kind, label, blurb, size, group: 'Addis study', fields: 'none', details: false,
  resources: () => [STUDY_PATH], render: data => <Chart meta={data.meta} />,
})
