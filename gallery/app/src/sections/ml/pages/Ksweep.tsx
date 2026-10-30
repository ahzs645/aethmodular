import type { SectionPageProps } from '../../types'
import { PagePending } from '../common'

/** IMPROVE validation is blind to k (ksweep_20261014): being built. */
export function Ksweep(_props: SectionPageProps) {
  return <PagePending id="ml-ksweep" title="IMPROVE validation is blind to k" notebooks="ksweep_20261014"
    plan="Addis drifts with the number of PLS factors; do smoke-like IMPROVE filters drift the same way, and does the standard IMPROVE validation see it?" />
}
