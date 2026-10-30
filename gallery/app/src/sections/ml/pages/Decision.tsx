import type { SectionPageProps } from '../../types'
import { PagePending } from '../common'

/** Decision workbook: where the reference model fails, specialists and the flag (decision_workbook_20261022 §3, §4, §6): being built. */
export function Decision(_props: SectionPageProps) {
  return <PagePending id="ml-decision" title="Decision workbook: where the reference model fails, specialists and the flag" notebooks="decision_workbook_20261022 §3, §4, §6"
    plan="Where the full calibration fails inside IMPROVE, whether specialist calibrations beat it, and which SPARTAN site-seasons the flag marks." />
}
