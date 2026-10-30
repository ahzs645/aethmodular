/**
 * The Machine learning section (src/sections/sections.json: private). The notebooks' model-fitting results:
 * component selection and cross-validation, protocols and transfer, the June 2026 calibration variants,
 * residual learners, target definitions, model forms and feature importance. Everything here needs a model
 * fitted on spectra, so it is precomputed (gallery/data/export_ml.py) and static: the pages draw the
 * notebooks' numbers and refit nothing. A public build swaps this module for ../omitted.ts.
 */
import { SectionShell } from '../SectionShell'
import type { Section, SectionPageProps } from '../types'
import { ML_PAGES } from './pages'

function MlPage(props: SectionPageProps) {
  return <SectionShell name="Machine learning" pages={ML_PAGES} props={props} />
}

const ml: Section = {
  key: 'ml',
  label: 'Machine learning',
  tabs: [{
    key: 'ml',
    label: 'Machine learning',
    title: 'The notebooks’ model-fitting results (PLS, component selection, cross-validation, residual learners), as they ran: static',
    pages: ML_PAGES.map(({ key, label, title, hidden }) => ({ key, label, title, hidden })),
    Page: MlPage,
  }],
  useFields: () => null,
}
export default ml
