import { ChartFrame, Note } from '@/components/ChartFrame'
import type { SectionPageProps } from '../../types'

/** Filter-only transfer limits (filter_only_diagnostics, filter_proportionality, filter_relationship_stability): being built. */
export function FilterOnly(_props: SectionPageProps) {
  return (
    <ChartFrame id="research-filter-only" title="Filter-only transfer limits" subtitle="How far the filter-only relationships carry across time: proportionality and relationship stability by period." exportable={false}>
      <Note>Notebooks: filter_only_diagnostics, filter_proportionality, filter_relationship_stability.</Note>
    </ChartFrame>
  )
}
