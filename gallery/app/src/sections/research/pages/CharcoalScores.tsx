import { ChartFrame, Note } from '@/components/ChartFrame'
import type { SectionPageProps } from '../../types'

/** Charcoal similarity per Addis filter (char_05–char_09): being built. */
export function CharcoalScores(_props: SectionPageProps) {
  return (
    <ChartFrame id="research-charcoal" title="Charcoal similarity per filter" subtitle="How closely each Addis spectrum matches the charcoal reference classes, joined to the filters so seasons, exclusions and loading apply." exportable={false}>
      <Note>Notebooks: char_05, char_06, char_07, char_08, char_09.</Note>
    </ChartFrame>
  )
}
