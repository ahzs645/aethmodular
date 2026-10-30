import type { SectionPageProps } from '../../types'
import { PagePending } from '../common'

/** Model form, every combination tried, and feature importance (ftir_54, ftir_55, ftir_51; ftir_07, ftir_08): being built. */
export function ModelForm(_props: SectionPageProps) {
  return <PagePending id="ml-model-form" title="Model form, every combination tried, and feature importance" notebooks="ftir_54, ftir_55, ftir_51; ftir_07, ftir_08"
    plan="PLS against other model forms on the same splits, the full combination search, and which wavenumbers the models lean on (VIP)." />
}
