import { ChartFrame, Note } from '@/components/ChartFrame'
import type { SectionPageProps } from '../../types'

/** Blank lines per deployed line (ftir_47, ftir_53 01): being built. */
export function BlankLines(_props: SectionPageProps) {
  return (
    <ChartFrame id="research-blank-lines" title="Blank lines per deployed line" subtitle="Per deployed HIPS calibration line: the field blanks’ R1–T1 relation, linear vs quadratic, and the Fabs each line implies." exportable={false}>
      <Note>Notebooks: ftir_47, ftir_53 01.</Note>
    </ChartFrame>
  )
}
