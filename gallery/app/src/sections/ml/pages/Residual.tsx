import type { SectionPageProps } from '../../types'
import { PagePending } from '../common'

/** Residual learners and null controls (ftir_43, ftir_45, ftir_46): being built. */
export function Residual(_props: SectionPageProps) {
  return <PagePending id="ml-residual" title="Residual learners and null controls" notebooks="ftir_43, ftir_45, ftir_46"
    plan="Whether a learner on the calibration residual finds anything a shuffled-label null does not, and what the paired increment says about mass." />
}
