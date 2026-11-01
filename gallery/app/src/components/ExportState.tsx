import type { ReactNode } from 'react'
import { ChartFrame, Empty } from '@/components/ChartFrame'

export interface LoadStateProps {
  /** what is loading, for the sentence ("the meeting follow-up export"); omitted: just "Loading…" */
  what?: string
  /** why it could not be read ('' or absent while loading) */
  error?: string
  /** shell command that rebuilds the export */
  command?: string
  /** the failure in the page's own words (where the export comes from, what it needs), in place of "Could not load …" */
  failed?: ReactNode
}

/**
 * The line a chart shows while its data loads, or when it cannot be read: inside a chart's own frame
 * (`<ChartFrame>…{data ? chart : <LoadState what="the profiles" error={error} />}`).
 */
export function LoadState({ what, error, command, failed }: LoadStateProps) {
  return <Empty>{error
    ? failed ?? <>Could not load {what ?? 'the data'}: {error}.{command && <> Run <code>{command}</code> and reload.</>}</>
    : <>Loading{what ? ` ${what}` : ''}…</>}</Empty>
}

/**
 * The panel a page shows while its export loads, or when it cannot be read: the same frame on every tab,
 * with the command that regenerates the file. With a title it is a titled chart frame; without, a plain
 * panel (a page or section body that has nothing to title yet).
 */
export function ExportState({ title, ...state }: LoadStateProps & {
  /** panel title, usually the page or section name */
  title?: string
}) {
  return title
    ? <ChartFrame title={title} exportable={false}><LoadState {...state} /></ChartFrame>
    : <section className="panel"><LoadState {...state} /></section>
}
