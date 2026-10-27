import { ChartFrame } from '@/components/ChartFrame'
import type { SectionPageProps } from '../../types'
import { RESEARCH_PAGES } from '../pages'

/** The section's first page: what each diagnostic page holds and the notebooks it reproduces. */
export function Index(_props: SectionPageProps) {
  const rest = RESEARCH_PAGES.filter((p) => p.key !== 'index')
  return (
    <ChartFrame id="research-index" title="Research diagnostics — notebook analyses brought into the gallery"
      subtitle="Analyses the gallery had the data for but not the notebook's own method. This section is private: a public build leaves it out (src/sections/sections.json)." exportable={false}>
      {rest.length === 0 ? <p className="frame-empty">No diagnostic pages yet.</p> : (
        <ul className="subtitle">{rest.map((p) => <li key={p.key}><strong>{p.label}</strong>{p.title ? ` — ${p.title}` : ''}</li>)}</ul>
      )}
    </ChartFrame>
  )
}
