import type { ComponentType } from 'react'
import { AreaHeader } from '@/components/AreaHeader'
import { defaultPage, type SectionPage, type SectionPageProps } from './types'

/** A flagged section's tab: the shared area heading with its sub-page picker (URL `aep`), and the open sub-page. */
export function SectionShell({ name, pages, props }: {
  name: string
  pages: (SectionPage & { Page: ComponentType<SectionPageProps> })[]
  props: SectionPageProps
}) {
  const current = pages.find((p) => p.key === props.page) ?? pages.find((p) => p.key === defaultPage(pages))!
  const Page = current.Page
  return (
    <>
      {/* a hidden page (an index) joins the picker only while it is open */}
      <AreaHeader area={name} pages={pages.filter((p) => !p.hidden || p.key === current.key)} value={current.key} onPage={props.onPage} />
      <Page {...props} />
    </>
  )
}
