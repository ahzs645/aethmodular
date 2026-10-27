import type { ComponentType } from 'react'
import { Segmented } from '@/components/ChartFrame'
import type { SectionPage, SectionPageProps } from './types'

/** A flagged section's tab: a heading, the sub-page picker (URL `aep`), and the open sub-page. */
export function SectionShell({ name, pages, props }: {
  name: string
  pages: (SectionPage & { Page: ComponentType<SectionPageProps> })[]
  props: SectionPageProps
}) {
  const current = pages.find((p) => p.key === props.page) ?? pages[0]
  const Page = current.Page
  return (
    <>
      <div className="panel weather-head">
        <div>
          <h2 style={{ margin: 0 }}>{name} · {current.label}</h2>
          {current.title && <p className="subtitle" style={{ margin: '4px 0 0' }}>{current.title}</p>}
        </div>
        {pages.length > 1 && <div className="weather-head-controls">
          <Segmented label="Page" value={current.key} options={pages.map((p) => p.key)} onChange={props.onPage}
            optionLabel={(k) => pages.find((p) => p.key === k)?.label ?? k} />
        </div>}
      </div>
      <Page {...props} />
    </>
  )
}
