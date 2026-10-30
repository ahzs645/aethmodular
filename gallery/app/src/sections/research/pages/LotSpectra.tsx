import { ChartFrame, Note } from '@/components/ChartFrame'
import type { SectionPageProps } from '../../types'

/** PTFE lot comparison for any site or sites (char_12 §4): being built. */
export function LotSpectra(_props: SectionPageProps) {
  return (
    <ChartFrame id="research-lot-spectra" title="PTFE lot comparison" subtitle="Spectra by filter lot for the sites in view, all filters and season-controlled, raw, second-derivative and AIRSpec-baselined." exportable={false}>
      <Note>Notebook: char_12 §4 (04_lot_effect, 04b_lot_airspec_view), extended to every site.</Note>
    </ChartFrame>
  )
}
