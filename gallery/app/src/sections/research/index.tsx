/**
 * The Research diagnostics section (src/sections/sections.json: private). Notebook analyses that the
 * gallery had the data for but not the method: HIPS/meteorology diagnostics, spectral and calibration
 * diagnostics. A public build swaps this module for ../omitted.ts.
 */
import { SectionShell } from '../SectionShell'
import type { Section, SectionPageProps } from '../types'
import { RESEARCH_PAGES } from './pages'
import './baselineLabDashboard'
import history from './history.generated.json'
import type { InvestigationHistory } from '@/lib/investigationHistory'

function ResearchPage(props: SectionPageProps) {
  return <SectionShell name="Research diagnostics" pages={RESEARCH_PAGES} props={props} />
}

const research: Section = {
  key: 'research',
  label: 'Research',
  history: history as InvestigationHistory,
  tabs: [{
    key: 'research',
    label: 'Research diagnostics',
    title: 'Notebook analyses brought into the gallery: HIPS, meteorology, spectral and calibration diagnostics',
    pages: RESEARCH_PAGES.map(({ key, label, title, hidden, samples }) => ({ key, label, title, hidden, samples })),
    Page: ResearchPage,
  }],
  useFields: () => null,
}
export default research
