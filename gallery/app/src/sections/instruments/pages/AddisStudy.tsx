import type { SectionPageProps } from '../../types'
import { AddisStudyBenchmarks, AddisStudyReadiness } from '../study/Context'
import { AddisStudyTimeline } from '../study/Timeline'
import { AddisStudySeasons } from '../study/Seasonal'
import { AddisStudyDiurnal } from '../study/Diurnal'
import { AddisStudyEvents } from '../study/Events'

/** Published study context and the saved local record, with a fixed study population. */
export function AddisStudy({ meta }: SectionPageProps) {
  return <>
    <section className="panel">
      <h2 style={{ margin: 0 }}>Addis study · published context and saved observations</h2>
      <p className="subtitle">
        <a href="https://doi.org/10.1021/acsestair.6c00140" target="_blank" rel="noreferrer">Shlipak et al. (2026)</a> follows Central and Jacros from April 2022 through March 2025.
        {' '}The gallery can currently show a shorter saved Jacros record, its coverage, seasonal patterns and festival windows. Central and local source-model inputs remain pending.
      </p>
      <p className="subtitle"><b>Fixed hourly study population.</b> These summaries use the saved Jacros hours and the calendar controls on this page.
        {' '}The Samples panel’s filter selections and mass-per-filter display do not change them. This is descriptive context; exact paper reproduction and independent validation are not established.</p>
    </section>
    <AddisStudyBenchmarks />
    <AddisStudyTimeline meta={meta} />
    <AddisStudySeasons meta={meta} />
    <AddisStudyDiurnal meta={meta} />
    <AddisStudyEvents meta={meta} />
    <AddisStudyReadiness />
  </>
}
