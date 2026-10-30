import type { SectionPageProps } from '../../types'
import { PagePending } from '../common'

/** Component selection and cross-validation schemes (ftir_11, ftir_20, ftir_23, ftir_32): being built. */
export function Components(_props: SectionPageProps) {
  return <PagePending id="ml-components" title="Component selection and cross-validation schemes" notebooks="ftir_11, ftir_20, ftir_23, ftir_32"
    plan="How many PLS factors each calibration keeps, under site-grouped and interleaved cross-validation, and how much the choice moves Addis." />
}
