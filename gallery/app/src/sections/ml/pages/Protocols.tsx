import type { SectionPageProps } from '../../types'
import { PagePending } from '../common'

/** Protocols end to end, transfer and bootstrap uncertainty (ftir_13, ftir_15, ftir_17, ftir_18, ftir_21, ftir_22, ftir_31): being built. */
export function Protocols(_props: SectionPageProps) {
  return <PagePending id="ml-protocols" title="Protocols end to end, transfer and bootstrap uncertainty" notebooks="ftir_13, ftir_15, ftir_17, ftir_18, ftir_21, ftir_22, ftir_31"
    plan="The interleaved (app) protocol against site-held-out validation end to end, the training-set transfer roundup, and bootstrap intervals of refits." />
}
