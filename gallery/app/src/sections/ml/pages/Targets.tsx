import type { SectionPageProps } from '../../types'
import { PagePending } from '../common'

/** What counts as EC: target definitions and Adama (ftir_41, ftir_42, ftir_44, ftir_48): being built. */
export function Targets(_props: SectionPageProps) {
  return <PagePending id="ml-targets" title="What counts as EC: target definitions and Adama" notebooks="ftir_41, ftir_42, ftir_44, ftir_48"
    plan="Calibrations trained on different EC definitions (TOR conventions, pyrolysis split, summed vs direct TC) and the Adama TOR filters through the locked calibrations." />
}
