import type { ReactNode } from 'react'
import { ChartFrame, Note } from '@/components/ChartFrame'

/**
 * A study's "short answers": a numbered list of findings in an unexported frame, with an optional note under it (the
 * run it comes from, what the numbers do not say). One look for every research page's findings panel.
 *
 *   <Findings id="xa-findings" title="… — short answers" note={<>{runLine(d)}. …</>}>
 *     <li><strong>…</strong> …</li>
 *   </Findings>
 */
export function Findings({ id, title, source, note, children }: {
  id?: string
  title: string
  /** the source line, as on any ChartFrame */
  source?: string
  /** a note under the list (a Note) */
  note?: ReactNode
  /** the findings, as <li> elements */
  children: ReactNode
}) {
  return (
    <ChartFrame id={id} source={source} title={title} exportable={false}>
      <ol style={{ margin: '4px 0 0 18px', lineHeight: 1.55 }}>{children}</ol>
      {note != null && <Note>{note}</Note>}
    </ChartFrame>
  )
}
