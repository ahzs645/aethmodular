import type { ReactNode } from 'react'
import { Note } from '@/components/ChartFrame'

/**
 * A study's "short answers": the numbered findings and an optional note under them (the run they come from, what the
 * numbers do not say), inside the panel's own unexported ChartFrame. One look for every research page's findings.
 *
 *   <FindingsList note={<>{runLine(d)}. …</>}>
 *     <li><strong>…</strong> …</li>
 *   </FindingsList>
 *
 * The frame stays in the page (an unexported ChartFrame with the panel's id and title), so the ⌘K index finds it.
 */
export function FindingsList({ note, children }: {
  /** a note under the list (a Note) */
  note?: ReactNode
  /** the findings, as <li> elements */
  children: ReactNode
}) {
  return <>
    <ol style={{ margin: '4px 0 0 18px', lineHeight: 1.55 }}>{children}</ol>
    {note != null && <Note>{note}</Note>}
  </>
}
