import { ChartFrame, Note } from '@/components/ChartFrame'
import type { SectionPageProps } from '../../types'

/** The mass lead (ftir_46): being built. */
export function MassLead(_props: SectionPageProps) {
  return (
    <ChartFrame id="research-mass-lead" title="The mass lead" subtitle="Does PM2.5 mass explain the calibration residual at fixed prediction? Partial slopes per site, computed live." exportable={false}>
      <Note>Notebooks: ftir_46.</Note>
    </ChartFrame>
  )
}
